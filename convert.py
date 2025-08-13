import torch
from safetensors.torch import save_file
import os
import argparse
from pathlib import Path

# ====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to safetensors with precision control")
    parser.add_argument("--input_model", type=str, required=True, help="Path to input PyTorch model (.pth, .pt or .ckpt)")
    parser.add_argument("--output_model", type=str, help="Path to output safetensors model (.safetensors)")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32",
                        help="Precision to convert model to (fp32, fp16, or bf16)")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args()

# ====================================
def get_tensor_info(state_dict):
    """Get information about tensors in the state dict"""
    total_params = 0
    dtypes = set()
    
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            total_params += value.numel()
            dtypes.add(str(value.dtype))
    
    return total_params, dtypes

# ====================================
def convert_model(input_path, output_path, precision, force_cpu=False, verbose=False):
    # Input validation
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine device
    if force_cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda:0"
    
    print(f"Using device: {device}")
    
    if verbose:
        print(f"Loading model from {input_path}...")
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"Input file size: {file_size_mb:.1f} MB")

    # Check file extension
    _, ext = os.path.splitext(input_path)
    if ext.lower() not in ['.pth', '.pt', '.ckpt']:
        print(f"Warning: Input file extension '{ext}' is not a common PyTorch format (.pth, .pt, .ckpt)")
        print("Attempting to load anyway...")

    try:
        # Load with weights_only=True for security (PyTorch 1.13+)
        if hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames:
            state_dict = torch.load(input_path, map_location=device, weights_only=True)
        else:
            state_dict = torch.load(input_path, map_location=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying without weights_only flag...")
        state_dict = torch.load(input_path, map_location=device)
    
    # Handle different checkpoint formats
    original_keys = list(state_dict.keys())
    if "state_dict" in state_dict:
        print("Detected checkpoint format with wrapped state_dict")
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        print("Detected checkpoint format with 'model' key")
        state_dict = state_dict["model"]
    elif "ema" in state_dict:
        print("Detected EMA checkpoint format")
        state_dict = state_dict["ema"]
    elif len(original_keys) > 0 and not any(isinstance(v, torch.Tensor) for v in state_dict.values()):
        # Check if we need to go one level deeper
        for key in original_keys[:3]:  # Check first few keys
            if isinstance(state_dict[key], dict) and any(isinstance(v, torch.Tensor) for v in state_dict[key].values()):
                print(f"Found nested state dict under key: {key}")
                state_dict = state_dict[key]
                break

    if verbose:
        total_params, dtypes = get_tensor_info(state_dict)
        print(f"Total parameters: {total_params:,}")
        print(f"Original dtypes: {', '.join(dtypes)}")

    print(f"Converting to {precision}...")
    converted_count = 0
    
    for key in state_dict:
        if isinstance(state_dict[key], torch.Tensor):
            original_dtype = state_dict[key].dtype
            if precision == "fp16":
                state_dict[key] = state_dict[key].half()
            elif precision == "bf16":
                state_dict[key] = state_dict[key].to(torch.bfloat16)
            elif precision == "fp32":
                state_dict[key] = state_dict[key].float()
            
            if verbose and state_dict[key].dtype != original_dtype:
                converted_count += 1
    
    if verbose:
        print(f"Converted {converted_count} tensors")
    
    # Generate output path if not provided
    if output_path is None:
        base_name = Path(input_path).stem
        output_dir = Path(input_path).parent
        output_path = output_dir / f"{base_name}_{precision}.safetensors"
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Move to CPU before saving (safetensors saves from CPU)
    if device != "cpu":
        print("Moving tensors to CPU for saving...")
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].cpu()
    
    print(f"Saving to {output_path}...")
    save_file(state_dict, str(output_path))
    
    if verbose:
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Output file size: {output_size_mb:.1f} MB")
    
    print("Conversion complete!")
    return str(output_path)

# ====================================
if __name__ == "__main__":
    try:
        args = parse_args()
        convert_model(args.input_model, args.output_model, args.precision, 
                     args.force_cpu, args.verbose)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

# ====================================
"""
# Basic usage (default fp32)
python convert_model.py --input_model model.pth

# Specify output path
python convert_model.py --input_model model.pth --output_model converted_model.safetensors

# Convert to half precision (fp16)
python convert_model.py --input_model model.pth --precision fp16

# Convert to bfloat16 precision
python convert_model.py --input_model model.pth --precision bf16

# Force CPU usage and verbose output
python convert_model.py --input_model model.pth --force_cpu --verbose

# Handle SUPIR checkpoint specifically
python convert_model.py --input_model SUPIR-v0F.ckpt --precision fp16 --verbose
"""
