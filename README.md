# Pth-2-Safetensors
A script to convert PyTorch model (.pth, .pt, ckpt) to safetensors format.

Installation
```bash
git clone https://github.com/yushan777/Pth-2-Safetensors.git
cd Pth-2-Safetensors

# Create virtual env (not absolutely necessary but it's best practice
# so you don't interfere with any system-wide dependencies)
python -m venv venv

# Activate venv
# For Linux, macOS  
source venv/bin/activate

# For Windows
venv\Scripts\activate.bat 

# Install a couple of dependencies
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
