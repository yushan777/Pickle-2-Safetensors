# Pickle-2-Safetensors
A script to convert PyTorch model (.pth, .pt, ckpt) to safetensors format.

Installation
```bash
# Clone this repo and go to the directory
git clone https://github.com/yushan777/Pickle-2-Safetensors.git
cd Pickle-2-Safetensors

# Create virtual env (not absolutely necessary but it's best practice
# so you don't interfere with any system-wide dependencies)
python3 -m venv venv

# Activate venv
# For Linux, macOS  
source venv/bin/activate

# For Windows
# venv\Scripts\activate.bat 

# Install a couple of dependencies
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128
pip install safetensors>=0.6.2
pip install packaging>=25.0
```

Usage
```
# Defaults to fp32
python3 convert.py --input_model model.pth

# Specify output path
python3 convert.py --input_model model.pth --output_model converted_model.safetensors

# Convert to half precision (fp16)
python3 convert.py --input_model model.pth --precision fp16

# Convert to bfloat16 precision
python3 convert.py --input_model model.pth --precision bf16
```
