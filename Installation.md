# Installation Guide

This guide provides step-by-step instructions to set up the project environment on a Raspberry Pi 4B.

## 1. System Infomation Check

Before starting, verify the system information and available resources.

```bash
# Check the available disk space (ensure you have at least a few GBs free)
df -h /

# Check the python version in global installation
python -V

# Check the OS release version on Pi 4B
cat /etc/os-release
```

## 2. Clone the Repository

Clone the project into your workspace and navigate into it.

```bash
git clone git@github.com:leputz/Pi4B_ONNX_Sudoku.git 
cd Pi4B_ONNX_Sudoku
```

## 3. Install `pyenv` and `pyenv-virtualenv`

We use `pyenv` to manage a specific Python version (3.10) without affecting the system's default Python.

```bash
# Install pyenv
curl https://pyenv.run | bash
```

**Configure your shell:**
Add the following lines to your `~/.bashrc` file (you can use `nano ~/.bashrc` to edit it):

```bash
# Load pyenv automatically by appending the following to 
# ~/.bash_profile if it exists, otherwise ~/.profile (for login shells)
# and ~/.bashrc (for interactive shells):

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# Load pyenv-virtualenv automatically
eval "$(pyenv virtualenv-init -)"
```

**Apply the changes:**
```bash
source ~/.bashrc
```

## 4. Install Python 3.10.13

First, install the necessary system libraries required to compile Python from source.

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

Next, use `pyenv` to install Python 3.10.13.

```bash
pyenv install 3.10.13
```

## 5. Set Up Virtual Environment

Create a dedicated virtual environment for this project and configure it to activate automatically when you enter the project directory.

```bash
# Create the virtual environment named "venv" based on Python 3.10.13
pyenv virtualenv 3.10.13 venv

# Enable automatic virtual environment activation in this directory
pyenv local venv
```

## 6. Install Project Dependencies

Ensure your virtual environment is active (you should see `(venv)` in your terminal prompt).

### ONNX Toolchain
```bash
pip install onnx onnxruntime onnxscript netron numpy

# (Optional) Upgrade pip to the latest version
python -m pip install --upgrade pip
```

### PyTorch (CPU build)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Data Generation & Processing Dependencies
```bash
pip install pandas pyarrow opencv-python tqdm
```

## 7. Verify the Installation

Test the toolchain from model definition, ONNX export, to running the model in ONNX Runtime.

```bash
python3 SmallNet.py 
```

## 8. Download the Dataset

Download the Sudoku dataset from Hugging Face to the project directory.

```bash
wget https://huggingface.co/datasets/Ritvik19/Sudoku-Dataset/resolve/main/train_1.parquet
```

---

## 🚀 Next Steps

Now that the environment and dataset are ready, it is time to run the data preparation scripts for performance benchmarking:

```bash
python3 PrepareData.py
# or v2, v3, v4...
```