# Pth-2-Safetensors
A script to convert PyTorch model to Safetensors format.

Dependencies
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu126
pip install safetensors
```

Usage
```
# Defaults to fp32
python convert_model.py --input_model model.pth

# Specify output path
python convert_model.py --input_model model.pth --output_model converted_model.safetensors

# Convert to half precision (fp16)
python convert_model.py --input_model model.pth --precision fp16

# Convert to bfloat16 precision
python convert_model.py --input_model model.pth --precision bf16
```
