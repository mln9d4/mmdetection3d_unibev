import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmdet.models.utils import PLUGIN_LAYERS
from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class CamToLidarAdapter(nn.Module):
    """A learnable mapping module that maps camera BEV features to lidar BEV space.
    
    Designed to work with frozen UniBEV backbone - the adapter runs inference through
    UniBEV to get features, then trains only this mapping module to predict where 
    lidar features should be based on camera features.
    
    Args:
        in_channels (int): Number of input channels (typically 256 for UniBEV)
        out_channels (int): Number of output channels. Default: same as in_channels
        hidden_channels (int): Number of hidden channels in MLP. Default: in_channels * 2
        bev_h (int): Height of BEV feature map. Default: 200
        bev_w (int): Width of BEV feature map. Default: 200
        num_layers (int): Number of convolutional layers. Default: 3
        with_residual (bool): Whether to use residual connections. Default: True
        with_norm (bool): Whether to use batch normalization. Default: True
        conv_cfg (dict): Config dict for convolution. Default: None
        norm_cfg (dict): Config dict for normalization. Default: dict(type='BN')
        activation (str): Activation function. Default: 'relu'
    """
    
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 hidden_channels=None,
                 bev_h=200,
                 bev_w=200,
                 num_layers=3,
                 with_residual=True,
                 with_norm=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 activation='relu'):
        super(CamToLidarAdapter, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels * 2
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_layers = num_layers
        self.with_residual = with_residual
        self.with_norm = with_norm
        self.activation = activation
        
        # Build convolutional layers to learn spatial-channel transformation
        self.conv_layers = nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels
        self.conv_layers.append(
            ConvModule(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if with_norm else None,
                act_cfg=dict(type=activation.upper() if activation else None) if activation else None
            )
        )
        
        # Middle layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                ConvModule(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if with_norm else None,
                    act_cfg=dict(type=activation.upper() if activation else None) if activation else None
                )
            )
        
        # Last layer: hidden_channels -> out_channels
        self.conv_layers.append(
            ConvModule(
                hidden_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if with_norm else None,
                act_cfg=None  # No activation on output
            )
        )
        
        # Projection layer for residual connection if dimensions don't match
        if self.with_residual and in_channels != out_channels:
            self.residual_proj = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None
            )
        else:
            self.residual_proj = None
    
    def forward(self, img_bev_embed):
        """Map camera BEV features to lidar BEV feature space.
        
        Args:
            img_bev_embed (Tensor): Camera BEV features with shape
                (batch_size, channels, bev_h, bev_w) or 
                (batch_size, bev_h*bev_w, channels)
        
        Returns:
            Tensor: Mapped features in lidar feature space with same shape as output
        """
        # Handle both flattened and spatial feature formats
        is_flattened = img_bev_embed.dim() == 3
        if is_flattened:
            # Reshape from (bs, hw, c) to (bs, c, h, w)
            batch_size = img_bev_embed.shape[0]
            img_bev_embed = img_bev_embed.reshape(
                batch_size, self.bev_h, self.bev_w, -1
            ).permute(0, 3, 1, 2).contiguous()
        
        # Save for residual connection
        identity = img_bev_embed if self.with_residual else None
        
        # Forward through convolutional layers
        out = img_bev_embed
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        
        # Add residual connection
        if self.with_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity
        
        # Reshape back to flattened format if input was flattened
        if is_flattened:
            batch_size = out.shape[0]
            out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, self.out_channels)
        
        return out


@NECKS.register_module()
class CamToLidarAdapterWithDistillation(nn.Module):
    """Mapping module with integrated distillation loss for frozen UniBEV training.
    
    This module is designed to work with a frozen UniBEV model:
    1. UniBEV runs in eval mode and produces img_bev_embed and pts_bev_embed
    2. This adapter maps img_bev_embed to predict where pts_bev_embed should be
    3. Only this module's weights are updated during training
    4. Gradients don't flow back to UniBEV
    
    Args:
        in_channels (int): Input feature channels
        adapter_cfg (dict): Config for the CamToLidarAdapter
        loss_type (str): Type of distillation loss. Options: 'mse', 'kl', 'cosine'. Default: 'mse'
        temperature (float): Temperature for KL divergence. Default: 1.0
        loss_weight (float): Weight for the distillation loss. Default: 1.0
    """
    
    def __init__(self,
                 in_channels,
                 adapter_cfg=None,
                 loss_type='mse',
                 temperature=1.0,
                 loss_weight=1.0):
        super(CamToLidarAdapterWithDistillation, self).__init__()
        
        if adapter_cfg is None:
            adapter_cfg = {}
        
        # Ensure in_channels is set in adapter config
        adapter_cfg.setdefault('in_channels', in_channels)
        
        self.adapter = CamToLidarAdapter(**adapter_cfg)
        self.loss_type = loss_type
        self.temperature = temperature
        self.loss_weight = loss_weight
    
    def forward(self, img_bev_embed, pts_bev_embed=None, return_loss=True):
        """Forward pass with optional distillation loss.
        
        Args:
            img_bev_embed (Tensor): Camera BEV features (will have no_grad applied)
            pts_bev_embed (Tensor, optional): Actual lidar BEV features for loss calculation
            return_loss (bool): Whether to calculate and return loss. Default: True
        
        Returns:
            dict or Tensor: 
                - If return_loss=True and pts_bev_embed provided: dict with 'mapped_features' and 'loss_distill'
                - If return_loss=False or pts_bev_embed=None: just the mapped tensor
        """
        # Detach camera features to prevent gradient flow to UniBEV (frozen backbone)
        img_bev_embed_detached = img_bev_embed.detach()
        
        # Map camera features to lidar space
        mapped_features = self.adapter(img_bev_embed_detached)
        
        # Return only mapped features if no loss calculation needed
        if not return_loss or pts_bev_embed is None:
            return mapped_features
        
        # Detach target features to prevent gradient flow
        pts_bev_embed_detached = pts_bev_embed.detach()
        
        # Calculate distillation loss
        loss_distill = self.calculate_distillation_loss(
            mapped_features, pts_bev_embed_detached
        )
        
        return {
            'mapped_features': mapped_features,
            'loss_distill': loss_distill * self.loss_weight
        }
    
    def calculate_distillation_loss(self, pred_feats, target_feats):
        """Calculate distillation loss between predicted and target features.
        
        Args:
            pred_feats (Tensor): Predicted features (camera mapped to lidar space)
            target_feats (Tensor): Target lidar features (detached)
        
        Returns:
            Tensor: Scalar distillation loss
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred_feats, target_feats)
        
        elif self.loss_type == 'kl':
            # KL divergence requires probability distributions
            # Normalize features to approximate distributions
            pred_probs = F.softmax(pred_feats / self.temperature, dim=-1)
            target_probs = F.softmax(target_feats / self.temperature, dim=-1)
            loss = F.kl_div(
                torch.log(pred_probs + 1e-8),
                target_probs,
                reduction='batchmean'
            )
        
        elif self.loss_type == 'cosine':
            # Cosine similarity loss (averaged across spatial and batch dimensions)
            # Flatten all but last dimension
            pred_flat = pred_feats.reshape(-1, pred_feats.shape[-1])
            target_flat = target_feats.reshape(-1, target_feats.shape[-1])
            loss = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


@NECKS.register_module()
class CamToLidarAdapterTrainer:
    """Utility class to manage training with frozen UniBEV and trainable adapter.
    
    Usage:
        trainer = CamToLidarAdapterTrainer(
            unibev_model=model,
            adapter_module=cam_to_lidar_adapter,
            adapter_optimizer=optimizer
        )
        
        # During training
        trainer.freeze_unibev()
        
        # In training loop
        loss = trainer.training_step(batch)
        loss.backward()
        trainer.adapter_optimizer.step()
    """
    
    def __init__(self, unibev_model, adapter_module, adapter_optimizer):
        self.unibev_model = unibev_model
        self.adapter_module = adapter_module
        self.adapter_optimizer = adapter_optimizer
    
    def freeze_unibev(self):
        """Freeze all UniBEV parameters."""
        for param in self.unibev_model.parameters():
            param.requires_grad = False
        self.unibev_model.eval()
        print("UniBEV model frozen - only adapter will be trained")
    
    def unfreeze_unibev(self):
        """Unfreeze UniBEV parameters."""
        for param in self.unibev_model.parameters():
            param.requires_grad = True
        self.unibev_model.train()
        print("UniBEV model unfrozen")
    
    def get_adapter_params(self):
        """Get only adapter parameters for optimization."""
        return self.adapter_module.parameters()
    
    def training_step(self, img_bev_embed, pts_bev_embed):
        """Single training step."""
        output = self.adapter_module(
            img_bev_embed, 
            pts_bev_embed, 
            return_loss=True
        )
        return output['loss_distill']