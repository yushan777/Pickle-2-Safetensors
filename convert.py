import torch
from safetensors.torch import save_file
import os
import argparse

# ====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to safetensors with precision control")
    parser.add_argument("--input_model", type=str, required=True, help="Path to input PyTorch model (.pth)")
    parser.add_argument("--output_model", type=str, help="Path to output safetensors model (.safetensors)")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32",
                        help="Precision to convert model to (fp32, fp16, or bf16)")
    return parser.parse_args()

# ====================================
def convert_model(input_path, output_path, precision):
    # Determine device - try CUDA:0 first, fall back to CPU if not available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model from {input_path}...")
    state_dict = torch.load(input_path, map_location=device)
    
    print(f"Converting to {precision}...")
    for key in state_dict:
        if isinstance(state_dict[key], torch.Tensor):
            if precision == "fp16":
                state_dict[key] = state_dict[key].half()
            elif precision == "bf16":
                state_dict[key] = state_dict[key].to(torch.bfloat16)
            elif precision == "fp32":
                state_dict[key] = state_dict[key].float()
    
    # If output path not provided, use input name but change extension
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_{precision}.safetensors"
    
    # Move to CPU before saving (safetensors typically saves from CPU)
    if device != "cpu":
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].cpu()
    
    print(f"Saving to {output_path}...")
    save_file(state_dict, output_path)
    print("Conversion complete!")

# ====================================
if __name__ == "__main__":
    args = parse_args()
    convert_model(args.input_model, args.output_model, args.precision)

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
"""