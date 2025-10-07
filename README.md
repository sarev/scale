# Installation

## NVIDIA CUDA Toolkit and cuDNN

- Download CUDA (e.g. 12.4) from NVIDIA website...
  - [cuda_12.4.1_551.78_windows.exe](https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
- Install CUDA toolkit by running the downloaded exe
- Download cuDNN for CUDA 12.4 from NVIDIA website...
  - [E.g. cudnn_9.1.0_windows.exe](https://developer.nvidia.com/rdp/cudnn-archive)
- Install cuDNN:
  - Create "C:\Program Files\NVIDIA\CUDNN\v9.1" directory
  - Copy cuDNN zipfile contents into that directory
- Download TensorRT from NVIDIA website...
  - [TensorRT-10.13.3.9.Windows.win10.cuda-12.9.zip](https://developer.nvidia.com/tensorrt/download)
- Install TensorRT:
  - Create "C:\Program Files\NVIDIA\TensorRT\v10.13.3.9"
  - Copy TensorRT zipfile contents into that directory

## Create and initialise a Python venv

```bash
cd scale
python -m venv .llm-venv
. .llm-venv/Scripts/activate
python -m pip install --upgrade pip wheel setuptools
```

## Run the CUDA activation script - bash version

```bash
. ./cuda-activate.sh     
```

## Install the CUDA-aware llama-cpp (large language models)

Precise version depends upon the host system:

    `nvcc --version`
 
should tell you the CUDA version you have (assuming you've installed
the CUDA toolkit and cuDNN from NVIDIA already). In my case, it's 12.4

```bash
pip install --no-cache-dir --force-reinstall https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp311-cp311-win_amd64.whl
```

## Tree-sitter (language parser framework)

```bash
pip install -U "tree-sitter==0.21.0" "tree-sitter-c==0.21.0" "tree-sitter-javascript==0.21.0"
```

## Optional (for downloading LLMs)...

```bash
pip install -U "huggingface_hub[cli]"
```
