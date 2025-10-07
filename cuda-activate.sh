#!/usr/bin/env bash
#
# Bash session-local environment for MSVC + CUDA + cuDNN on Windows
#
# This script is highly specific to my Windows 11 + NVIDIA RTX 4060 setup!

# Remove all PATH entries that contain the given substring or glob
path_rm_match() {
    local pat="$1"
    local IFS=:
    read -ra parts <<<"$PATH"
    local out=()
    for d in "${parts[@]}"; do
    [[ "$d" == *"$pat"* ]] || out+=("$d")    # glob match
    done
    PATH=$(IFS=:; printf '%s' "${out[*]}")
    export PATH
}

# Clean up PATH
echo
path_rm_match "/CUDA"
path_rm_match "\CUDA"
path_rm_match "/CUDNN"
path_rm_match "\CUDNN"
path_rm_match "/TensorRT"
path_rm_match "\TensorRT"
echo "Removed all CUDA, cuDNN and TensorRT from PATH"

# Visual Studio binaries...
VS=$(cygpath.exe -u "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64")

# NVIDIA CUDA toolkit...

# export CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
CUDA_HOME=$(cygpath.exe -u "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export CUDA_TOOLKIT_ROOT="$CUDA_HOME"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"

CB1="$CUDA_HOME/bin"

CL1="$CUDA_HOME/lib"
CL2="$CUDA_HOME/lib/x64"
CL3="$CUDA_HOME/libnvvp"
CL4="$CUDA_HOME/extras/CUPTI/lib64"

# NVIDIA cuDNN...

CUDNN_HOME=$(cygpath.exe -u "C:\Program Files\NVIDIA\CUDNN\v9.1")
# CUDNN_HOME=$(cygpath.exe -u "D:\Programming\Aurora v2\CUDNN\v9.1")

# CDNN="C:\Program Files\NVIDIA\CUDNN\v8.9\bin"
export CUDNN_HOME

UB1="$CUDNN_HOME/bin"
UB2="$CUDNN_HOME/bin/12.4"

UL1="$CUDNN_HOME/lib"
UL2="$CUDNN_HOME/lib/12.4/x64"

# NVIDIA TensorRT...

TRT_HOME=$(cygpath.exe -u "C:\Program Files\NVIDIA\TensorRT\v10.13.3.9")
# TRT_HOME=$(cygpath.exe -u "D:\Programming\Aurora v2\TensorRT\v10.13.3.9")
export TRT_HOME

TB1="$TRT_HOME/bin"

TL1="$TRT_HOME/lib"

# Assemble all the 'bin' and 'lib' paths

BINS="$CB1:$UB1:$UB2:$TB1"
LIBS="$CL1:$CL2:$CL3:$CL4:$UL1:$UL2:$TL1"

export CUDA_BIN_PATH="$CUDA_HOME:$CB1"
export CUDA_LIB_PATH="$CUDA_HOME:$CL1:$CL2:$CL3:$CL4"

export LD_LIBRARY_PATH="$LIBS:$LD_LIBRARY_PATH"

export PATH="$PATH:$BINS:$LIBS"

# Specify the CUDA architectures that you want PyTorch to build for when compiling CUDA kernels
export TORCH_CUDA_ARCH_LIST=8.6

echo
echo "Assuming NVIDIA RTX 4060 laptop GPU (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
echo
echo "CUDA_HOME = '$CUDA_HOME'"
echo "CUDNN_HOME = '$CUDNN_HOME'"
echo "TRT_HOME = '$TRT_HOME'"
echo

if [ ! -d "$CUDA_HOME" ]; then
    echo "CUDA installation missing!" >&2
fi

if [ ! -d "$CUDNN_HOME" ]; then
    echo "cuDNN installation missing!" >&2
fi

if [ ! -d "$TRT_HOME" ]; then
    echo "TensorRT installation missing!" >&2
fi
