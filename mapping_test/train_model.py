# import all necessary libraries

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import wandb
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from timeit import default_timer as timer 
from mmdet.models import HEADS
from mmdet3d.unibev_plugin.models.dense_heads import bev_consumer
import matplotlib.pyplot as plt
print("Libraries imported successfully")
# print(torch.__version__)

# Select Cuda Device to work on
torch.manual_seed(42)

device_number = "6"
device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# config = dict(
#     epochs=1000,
#     save_interval=20,
#     batch_size=1,
#     learning_rate=1e-3,
#     weight_decay=1e-2,
#     dataset="Nuscenes",
#     architecture="UNetAttention",
#     input_channels=256,
#     output_channels=256,
#     channel_sizes=[256, 256, 512, 512],
#     T_0=200,
#     T_mult=2,
#     eta_min=1e-6,
#     start_factor=0.1,
#     end_factor=1.0,
#     linear_lr_total_iters=400,
#     milestones=50,
#     )

config = dict(
    epochs=1000,
    save_interval=20,
    batch_size=1,
    learning_rate=1e-3,
    weight_decay=1e-2,
    dataset="Nuscenes",
    architecture="VanillaVAE",
    in_channels=512,
    output_channels=256,
    hidden_dim=1024,
    latent_dim=2048,
    T_0=200,
    T_mult=2,
    eta_min=1e-6,
    start_factor=0.1,
    end_factor=1.0,
    linear_lr_total_iters=400,
    milestones=50,
)

def save_loss_gradient_map(preds, targets, model, epoch, batch_idx):
    """
    Visualizes where the loss is pulling the model.
    Bright spots = high influence.
    """
    # 1. We need the gradient of the loss with respect to the prediction
    # We must ensure the prediction has grad enabled
    preds_for_grad = preds.detach().requires_grad_(True)
    
    # 2. Re-calculate loss for this specific pair
    loss_dict = model.loss(preds_for_grad, targets)
    loss = loss_dict['bev_consumer_loss_std_weighted_l1loss']
    
    # 3. Calculate gradients: dLoss / dPreds
    grad = torch.autograd.grad(loss, preds_for_grad)[0]
    
    # 4. Reshape to spatial (B, C, H, W) and take mean over channels
    bs, n, c = grad.shape
    grad_spatial = grad.transpose(1, 2).view(bs, c, 200, 200) # Using your bev_h/w
    grad_map = grad_spatial[0].cpu().numpy() # First sample in batch

    # 5. Plot with Robust Scaling (to avoid the "one color" issue)
    plt.figure(figsize=(6, 6))
    v_min, v_max = np.percentile(grad_map[0], [2, 98]) # Clip outliers
    
    plt.imshow(grad_map[0], cmap='magma', vmin=v_min, vmax=v_max)
    plt.colorbar(label="Gradient Intensity")
    plt.title(f"Loss Gradient Map (Spatial Influence)\nEpoch {epoch}")
    
    # Create the directory if it doesn't exist
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    save_path = os.path.join(save_dir, f"spatial_influence_ep{epoch}_b{batch_idx}.png")
    plt.savefig(save_path)
    plt.close()

def load_data():
    # Load the saved data
    all_img = torch.load('/home/mingdayang/mmdetection3d/mapping_test/all_img_bev.pt')
    all_pts = torch.load('/home/mingdayang/mmdetection3d/mapping_test/all_pts_bev.pt')

    X, y = all_img["img_bev_embed"], all_pts["pts_bev_embed"]

    return X, y

def create_splits(X, y, train_split=0.8):
    train_split = int(train_split * len(X)) # 80% of data used for training set, 20% for testing 
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    return X_train[:], y_train[:], X_test[:], y_test[:]

def make_loader(batch_size, x, y):
    tensor_set = TensorDataset(x, y)

    dataloader = DataLoader(tensor_set, batch_size=batch_size, shuffle=True)
        # Let's check out what we've created

    return dataloader

def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="manual-training-notebook", config=hyperparameters) as run:
        # access all HPs through run.config, so logging matches execution.
        config = run.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer, scheduler = make(config)

        # and use them to train the model
        train(model, train_loader, test_loader, criterion, optimizer, scheduler, config)
        save_model(model, config)

        return model

def make(config):
    # Make the data
    X, y = load_data()
    X_train, y_train, X_test, y_test = create_splits(X, y, train_split=0.8)
    print(f"Train data shape: {X_train.shape}, {y_train.shape}")
    train_loader = make_loader(config.batch_size, X_train, y_train)
    test_loader = make_loader(config.batch_size, X_test, y_test)

    # Make the model
    model = build_model(config)

    # Make the loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min)
    scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=config.start_factor, end_factor=config.end_factor, total_iters=config.linear_lr_total_iters)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                                    schedulers=[scheduler2, scheduler1], 
                                    milestones=[config.milestones],
                                )
    
    return model, train_loader, test_loader, criterion, optimizer, scheduler

def build_model(config):
    model_class = HEADS.get(config.architecture)
    input_channels=config.input_channels
    output_channels=config.output_channels
    channel_sizes=config.channel_sizes
    model = model_class(input_channels=input_channels, output_channels=output_channels, channel_sizes=channel_sizes)
    return model

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, config):
    run = wandb.init(project="manual-training-notebook", config=config)
    run.watch(model, criterion, log="all", log_freq=10)
    train_time_start_on_cpu = timer()
    model.to(device)
    for epoch in tqdm(range(config.epochs)):
        test_loss = 0.0
        train_loss =0.0

        model.train()

        print(f"Epoch {epoch}/{config.epochs}\n-------------------------------")

        for batch_idx, (X, y) in enumerate(train_loader):
            # Move data to device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            outputs = model(X)

            # 2. Compute loss
            # loss = criterion(outputs, y)
            loss = model.loss(outputs, y)
            loss = loss['bev_consumer_loss_std_weighted_l1loss'] + loss['bev_consumer_loss_MS_SSIM']
            train_loss += loss
            print(loss.requires_grad)
            # 3. Zero the gradients
            optimizer.zero_grad()

            # 4. Backward pass
            loss.backward() 

            if epoch % 10 == 0 and batch_idx % 20 == 0:
                save_loss_gradient_map(outputs, y, model, epoch=epoch, batch_idx=batch_idx)
                # save_activation_map(model, X, epoch=epoch)

            # 5. Optimizer step
            optimizer.step()
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            run.log({"learning_rate": current_lr})

            if batch_idx % 3 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)


        ### validation los
        model.eval()
        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                # loss = criterion(outputs, y)
                loss = model.loss(outputs, y)
                loss = loss['bev_consumer_loss_std_weighted_l1loss'] + loss['bev_consumer_loss_MS_SSIM']
                test_loss += loss

            avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")
        run.log({"train_loss": avg_train_loss, "test_loss": avg_test_loss, "epoch": epoch})

        if epoch % config.save_interval == 0:
            save_model(model, config, epoch)

    run.finish()
    train_time_end_on_cpu = timer()
    total_train_time_model = print_train_time(start=train_time_start_on_cpu, 
                                           end=train_time_end_on_cpu,
                                           device=str(next(model.parameters()).device))
    return model

def save_model(model, config, epoch=None):
    MODEL_PATH = os.path.join(os.getcwd(), 'models', config.architecture)
    os.makedirs(MODEL_PATH, exist_ok=True)
    if epoch is None:
        MODEL_SAVE_PATH = os.path.join(MODEL_PATH, f"{config.architecture}_model_last_epoch.pth")
    else:
        MODEL_SAVE_PATH = os.path.join(MODEL_PATH, f"{config.architecture}_model_epoch{epoch}.pth")

    temp_ = {}
    temp_['model_type'] = config.architecture
    temp_['model_config'] = {
        'input_channels': config.input_channels,
        'output_channels': config.output_channels,
        'channel_sizes': config.channel_sizes
    }
    temp_['state_dict'] = model.state_dict()

    torch.save(temp_, MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
def main():
    model = model_pipeline(config)




if __name__=="__main__":
    main()