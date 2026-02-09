"""Script to extract BEV consumer model from UniBEV checkpoint and save as standalone model.

This script:
1. Loads a UniBEV training checkpoint
2. Extracts the bev_consumer type and config from metadata
3. Extracts bev_consumer weights from state_dict
4. Saves as a standalone checkpoint that can be loaded directly as a torch model

Usage:
    python export_bev_consumer.py --checkpoint path/to/checkpoint.pth --output path/to/output.pth
"""

import torch
import argparse
import os
from pathlib import Path


def extract_bev_consumer_config(config_dict):
    """Extract bev_consumer configuration from nested config dict.
    
    Args:
        config_dict: The config dictionary from checkpoint metadata
        
    Returns:
        tuple: (model_type, bev_consumer_config_dict)
    """
    # Navigate through nested config structure
    # Config structure: model -> pts_bbox_head -> bev_consumer
    
    if isinstance(config_dict, str):
        # If config is stored as string, we need to parse it
        # This happens with some mmdet configs
        import re
        import ast
        
        # Try to find bev_consumer dict pattern
        pattern = r"bev_consumer=dict\((.*?)\)(?:,|\))"
        match = re.search(pattern, config_dict, re.DOTALL)
        
        if match:
            consumer_str = match.group(1)
            
            # Parse the string into a dictionary
            config_params = {}
            
            # Extract type first, as it's essential
            type_match = re.search(r"type='([^']+)'", consumer_str)
            if type_match:
                model_type = type_match.group(1)
            else:
                raise ValueError("Could not find 'type' in bev_consumer config")

            # A more robust way to parse the config string is to split it
            # by commas that are not inside parentheses or brackets.
            
            # Regex to split by comma, but not inside brackets or parentheses
            # This is a simplified parser.
            pairs = re.split(r',(?![^\(]*\))(?![^\[]*\])', consumer_str)
            
            for pair in pairs:
                if '=' not in pair:
                    continue
                
                key, value_str = pair.split('=', 1)
                key = key.strip()
                value_str = value_str.strip()
                
                if key == 'type':
                    continue
                
                # Handle `dict(...)` by recursively parsing
                if value_str.startswith('dict(') and value_str.endswith(')'):
                    inner_content = value_str[5:-1]
                    # This is a simplified parser for nested dicts.
                    # It might not handle all edge cases.
                    inner_dict = {}
                    try:
                        # A more robust way to parse dict content by converting it to a valid python dict string.
                        # e.g. "key1=value1, key2=value2" -> "{'key1':value1, 'key2':value2}"
                        
                        # First, add quotes around keys
                        processed_content = re.sub(r'(\w+)=', r"'\1':", inner_content)
                        
                        # ast.literal_eval can handle numbers, booleans, None, and strings in quotes.
                        # It will fail on unquoted strings, which is what we want to avoid.
                        # The values in the config string seem to be numbers, so this might be enough.
                        
                        inner_dict = ast.literal_eval("{" + processed_content + "}")

                    except Exception as e:
                        print(f"Could not parse inner dict string with regex method: {inner_content}")
                        print(f"Error: {e}")
                        # Fallback to old method if the above fails
                        inner_pairs = re.split(r',(?![^\(]*\))(?![^\[]*\])', inner_content)
                        for inner_pair in inner_pairs:
                            if '=' in inner_pair:
                                inner_key, inner_val_str = inner_pair.split('=', 1)
                                inner_key = inner_key.strip()
                                inner_val_str = inner_val_str.strip()
                                try:
                                    inner_dict[inner_key] = ast.literal_eval(inner_val_str)
                                except (ValueError, SyntaxError):
                                    inner_dict[inner_key] = inner_val_str
                    config_params[key] = inner_dict
                else:
                    # For other values, try ast.literal_eval
                    try:
                        config_params[key] = ast.literal_eval(value_str)
                    except (ValueError, SyntaxError):
                        # If it fails, keep it as a string
                        config_params[key] = value_str
            
            return model_type, config_params
    
    elif isinstance(config_dict, dict):
        # Navigate dict structure
        model_cfg = config_dict.get('model', {})
        
        if isinstance(model_cfg, dict):
            pts_bbox_head = model_cfg.get('pts_bbox_head', {})
            
            if isinstance(pts_bbox_head, dict):
                bev_consumer = pts_bbox_head.get('bev_consumer', {})
                
                if isinstance(bev_consumer, dict):
                    model_type = bev_consumer.pop('type', None)
                    if model_type is None:
                        raise ValueError("No 'type' found in bev_consumer config")
                    
                    return model_type, bev_consumer
    
    raise ValueError("Could not find bev_consumer configuration in checkpoint metadata")


def extract_bev_consumer_state_dict(full_state_dict, prefix='pts_bbox_head.bev_consumer.'):
    """Extract bev_consumer parameters from full model state_dict.
    
    Args:
        full_state_dict: Complete model state_dict from checkpoint
        prefix: Prefix to filter bev_consumer parameters
        
    Returns:
        dict: Filtered state_dict with prefix removed
    """
    bev_consumer_state_dict = {}
    
    for key, value in full_state_dict.items():
        if key.startswith(prefix):
            # Remove the prefix to get clean parameter names
            new_key = key[len(prefix):]
            bev_consumer_state_dict[new_key] = value
    
    return bev_consumer_state_dict


def export_bev_consumer(checkpoint_path, output_path=None, verbose=True):
    """Main function to export bev_consumer from UniBEV checkpoint.
    
    Args:
        checkpoint_path: Path to UniBEV training checkpoint
        output_path: Path to save exported model (optional)
        verbose: Whether to print extraction details
        
    Returns:
        dict: Exported checkpoint dictionary
    """
    if verbose:
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if verbose:
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Extract config
    meta = checkpoint.get('meta', {})
    config = meta.get('config', {})
    
    if verbose:
        print(f"Meta keys: {list(meta.keys())}")
    
    # Extract bev_consumer configuration
    model_type, bev_consumer_config = extract_bev_consumer_config(config)
    
    if verbose:
        print(f"\n✓ Found BEV consumer type: {model_type}")
        print(f"✓ BEV consumer config: {bev_consumer_config}")
    
    # Extract state_dict
    full_state_dict = checkpoint.get('state_dict', {})
    bev_consumer_state_dict = extract_bev_consumer_state_dict(full_state_dict)
    
    if verbose:
        print(f"\n✓ Extracted {len(bev_consumer_state_dict)} parameters:")
        for key, value in list(bev_consumer_state_dict.items())[:5]:
            print(f"    {key}: {value.shape}")
        if len(bev_consumer_state_dict) > 5:
            print(f"    ... and {len(bev_consumer_state_dict) - 5} more parameters")
    
    # Create standalone checkpoint
    standalone_checkpoint = {
        'model_type': model_type,
        'model_config': bev_consumer_config,
        'state_dict': bev_consumer_state_dict,
        'meta': {
            'source_checkpoint': str(checkpoint_path),
            'extracted_from': 'pts_bbox_head.bev_consumer',
        }
    }
    
    # Add epoch info if available
    if 'epoch' in checkpoint:
        standalone_checkpoint['meta']['source_epoch'] = checkpoint['epoch']
    
    # Save if output path provided
    if output_path:
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        torch.save(standalone_checkpoint, output_path)
        
        if verbose:
            print(f"\n✓ Saved standalone model to: {output_path}")
            print(f"  Model type: {model_type}")
            print(f"  Parameters: {len(bev_consumer_state_dict)}")
            
            # Calculate file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  File size: {file_size:.2f} MB")
    
    return standalone_checkpoint


def load_standalone_model(checkpoint_path, device='cpu'):
    """Helper function to load the exported standalone model.
    
    Args:
        checkpoint_path: Path to exported standalone checkpoint
        device: Device to load model on
        
    Returns:
        tuple: (model_instance, model_config)
        
    Example:
        >>> model, config = load_standalone_model('bev_consumer.pth')
        >>> model.eval()
        >>> output = model(input_tensor)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_type = checkpoint['model_type']
    model_config = checkpoint['model_config']
    state_dict = checkpoint['state_dict']
    
    print(f"Loading {model_type} model")
    print(f"Config: {model_config}")
    
    # User needs to instantiate the model with proper imports
    print("\nTo load this model, you need to:")
    print(f"1. Import the model class: from mmdet.models import HEADS")
    print(f"2. Get the class: ModelClass = HEADS.get('{model_type}')")
    print(f"3. Instantiate with config parameters")
    print(f"4. Load state_dict: model.load_state_dict(checkpoint['state_dict'])")
    
    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export BEV consumer model from UniBEV checkpoint'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to UniBEV training checkpoint (e.g., latest.pth, epoch_X.pth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save exported model (default: same directory as checkpoint with _bev_consumer.pth suffix)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Generate default output path if not provided
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        output_name = checkpoint_path.stem + '_bev_consumer.pth'
        args.output = str(checkpoint_path.parent / output_name)
    
    # Export model
    export_bev_consumer(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        verbose=not args.quiet
    )
    
    print("\n" + "="*60)
    print("Export complete! To use this model:")
    print("="*60)
    print("""
# Load checkpoint
import torch
checkpoint = torch.load('path/to/exported_model.pth')

# Import and instantiate model
from mmdet.models import HEADS
ModelClass = HEADS.get(checkpoint['model_type'])
model = ModelClass(**checkpoint['model_config'])  # Use config params

# Load weights
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    output = model(input_tensor)
""")
