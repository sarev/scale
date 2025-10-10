# SCALE — Source Code Annotation with LLM Engine

SCALE is a local, scriptable tool that adds or updates code comments and docstrings using an on-device LLM loaded via `llama.cpp`. It is designed to slot into existing build or CI pipelines and to operate deterministically with tight output control.

While SCALE can take a lot of the pain out of creating and maintaining comments on a codebase, you should still review its output and potentially make an editorial pass - just like you would in a code review of a team mate's work.

![Illustration](./scale.png)

## What it Does

SCALE will ingest a source file, parse it to find routines (functions, methods, classes) and will create one
comment per routine, following the comment style template (as defined in "scale-cfg"). The output will include these comments but aims to avoid touching the code itself, thus there should be no functional/behaviour change to the program, only the main comments on each routine are touched.

You are encouraged to look at the templates in "scale-cfg" and to adjust them to suit your needs (style guides, etc.) and your specific codebase. The more the examples provided in these files align with the code you are likely to process, the more effective SCALE will be in useful comment generation. There is one template per supported language.

The "scale-cfg/comment.txt" file is the main 'system prompt' which SCALE passes into the LLM to guide its behaviour. You may also want to edit this, but in general this shouldn't be necessary.

## How it Works

Using Python as an example, SCALE will find every "class", "def", and "async def" in the input file and process
them from the most deeply-nested up. For example:

```python
class Foo:
  ...some class attributes...
  def __init__():
    ...some of __init__ body...
    def bah():
      def mad():
        ...more deeply nested function...
      ...nested function...
    ...remainder of __init__ body...
```

The commenting process will start by ingesting the whole source file and asking the LLM to create a summary description of it. This summary is provided to the LLM for each subsequent comment-generation turn. It is useful to give SCALE some understanding of the whole file, because it helps with comprehension of each function, class, and method as it is processing, but we just use the summary to help keep the context window small (i.e. to help performance).

Then, the parser looks for all imports in the program (if any) and outputs that list for the LLM, because this can provide critical contextual information (e.g. `import numpy as np` is important for knowing what `np` means when encountered in the code).

The 'conversation' that is maintained during this process is between the 'user' (the tool) and the 'assistant'
(the LLM). On each turn, the user asks the assistant to generate a comment for the block of code they have
supplied (a function, method, or class). These turns are not recorded, so the LLM isn't building an
ever-increasing context window - we discard each chunk once it's done. All the LLM has to go on is:

1. The system prompt
2. The summary of the program as a whole
3. The comment template prompt
4. A list of any imports/includes (or similar) found in the program
5. The function/method/class it's been asked to generate a comment for

In the Python example above, SCALE will extract the `mad()` function - its signature, any preceding decorators, and the body code - and write the docstring for it. If there was already a docstring following the signature, that will be updated/replaced.

Next, SCALE will move to `bah()`. It will keep the signature of `mad()` but just include the (new) docstring. The body code for `mad()` is discarded, as it is a potential distraction. In a way, the LLM is 'remembering' the nested function `mad()` because it has been given the docstring (it generated earlier) for it.

Once we have the docstring for `bah()`, we can move on to `__init__()`. Again, the body of `bah()` is replaced with just its docstring here and `mad()` is removed altogether. Thus, SCALE is only looking at the code directly within the scope of `__init__()` and the signatures and docstrings for any functions that are immediately nested within (and not any more deeply nested ones).

Finally, it reaches class `Foo`. It will look at the class signature (and any preceding decorators) and the signatures and docstrings of the methods - in this case, just `__init__()`. So the comment generated for the class will be based upon a collective assessment of what all the method docstrings say, plus whatever class attributes there might be (as well as any pre-existing docstring the class may have had).

Once all the docstrings have been generated, the original source code is patched to place them into the correct locations, replacing any pre-existing docstrings as required. This patching process ensures that the executable code, indentation, other comments and empty lines are all preserved without any risk of the LLM inadvertently editing them or hallucinating more code!

### Language Support

| Language   | Parser / approach            | Comments generated                                                  |
| ---------- | ---------------------------- | ------------------------------------------------------------------- |
| Python     | CPython `ast`                | Docstrings for classes/functions/methods; import list passed to LLM |
| JavaScript | Tree-sitter (ESM + CommonJS) | JSDoc for functions/classes; import/require list passed to LLM      |
| C          | Tree-sitter                  | Function comments; `#include` list passed to LLM                    |

### Line Endings and Encoding

SCALE detects the dominant line ending (LF/CR/CRLF), preserves trailing newlines, and writes exactly what you specify. Files are treated as UTF-8 with `surrogateescape` to preserve undecodable bytes during round-trip. This avoids platform newline translation and keeps arbitrary input bytes intact.

## Using SCALE (typical)

Install the dependencies, download an LLM in GGUF format and then:

```bash
# Generate comments/docstrings for a source file with a local GGUF model
python scale.py --model /path/to/model.gguf --comment /path/to/file.py --output /path/to/output.py --verbose

# Generate comments and update the file in place (short form command line switches)
python scale.py -m /path/to/model.gguf -c /path/to/file.py -o /path/to/file.py -v
```

Use `--help` for more guidance on the CLI interface.

You can override LLM features, such as the chat format detection (`--format`), temperature/top-p/top-k, repeat penalty, context size, batch size, and GPU offload layers to suit different models, hardware and quantisations.

## Extensibility

* Add new languages by implementing a `generate_language_comments` function matching the Python worker’s interface and wiring it in the dispatcher. 
* Extend or swap prompt templates by editing files under `scale-cfg` without changing code. 

## Current Scope and Limits

* The current implementation only supports Python, C and JavaScript source code.
* Quality and speed depend on the chosen model, GGUF quant, context size and GPU memory. The wrapper exposes these controls but does not auto-tune them to your system.
* This has been tested on a Windows 11 laptop with an NVIDIA RTX 4060 mobile GPU. Better hardware than this will be able to utilise stronger LLMs.

## Picking an LLM

Most LLMs are pretty huge! Several GiB up to several **hundred** GiB.

As a rule of thumb, if your GPU has 8 GiB of VRAM, you should use a model with around that many parameters or fewer **and** you will probably need it to be 'quantised'. If a model has 8 billion parameters (i.e. fairly small!) and uses 16-bit floats, that's two bytes per parameter (16 GiB). Quantising squashes all the float values into smaller representations, e.g. 8, 6 or even 4 bits, at the cost of accuracy (the model becomes a little bit dumber). For example, an 8 billion parameter model with 4-bit quantisation will only require 8 x 0.5 byte = 4 GiB of VRAM.

However, you _also_ need a chunk of VRAM to work with while the LLM is running, in addition to loading the model itself. Typically you should leave maybe 25% - 30% of your VRAM available for workspace. One reason is to hold the 'context window' for the model. The larger the context window (how much conversation the LLM can see at any given time) the more VRAM you need available in addition to the model - or things will spill to RAM and disc during execution, killing performance.

On my RTX 4060 laptop GPU with 8 GiB of RAM, I found that I could get away with an 8 billion parameter, 6 bit quantised LLM and have room for 12k tokens of context window. Much more than that, and my system would start thrashing to CPU/disc. Better GPUs would be able to load significantly larger (quantised) models and/or have much larger context windows.

# Installation Guide for Linux Users

Most Linux systems will come with Python 3.xx pre-installed. This guide assumes you are on a Debian-based OS and _haven't_ got Python installed yet. You have a recent NVIDIA GPU but haven't yet installed the drivers.

```bash
# Basic Python components
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential git cmake pkg-config

# Add contrib and non-free-firmware components
sudo sed -i 's/main$/main contrib non-free-firmware/' /etc/apt/sources.list
sudo apt update

# Install the packaged NVIDIA driver - skip to the virtual environment steps if you don't have a GPU
sudo apt install -y nvidia-driver

# Reboot to load the driver
sudo reboot

# Verify CUDA available (and which version you have)
nvidia-smi

# Create Python virtual environment
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip setuptools wheel

# Install Python deps used by SCALE
pip install -U "tree-sitter==0.21.0" "tree-sitter-c==0.21.0" "tree-sitter-javascript==0.21.0" huggingface-hub

# Install llama.cpp bindings...

# Use the wheel index that hosts CUDA-enabled builds (cuXXX is your CUDA version)
pip install -U "llama-cpp-python[cuda]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# If you can't find a matching wheel, you can compile with...
export CMAKE_ARGS="-DLLAMA_CUDA=on -DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1
pip install -U llama-cpp-python

# If you don't even have CUDA/NVIDIA GPU, use a CPU-only build (very slow!)
pip install -U llama-cpp-python

# Get SCALE. If you already have the repo contents in ./scale, skip the clone
git clone https://github.com/sarev/scale.git
cd scale

# Download a local LLM from https://huggingface.co/models?library=gguf&sort=trending

# Log in (opens a browser or takes a token)
huggingface-cli login

# Make a local models folder
mkdir -p models

# Download a quantised GGUF (example: Q4_K_M) - this can take a loooong time...
huggingface-cli download \
  bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --include "*Q6_K.gguf" \
  --local-dir models \
  --local-dir-use-symlinks False

MODEL_PATH="models/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"

# Example: annotate a Python file
python scale.py \
  --model "$MODEL_PATH" \
  --comment path/to/input.py \
  --output  out/annotated.py \
  --verbose
```

# Installation Guide for Windows Users

I've included a lot of optional steps below. Skip over them if you just want to try SCALE out - these are only really useful for people who are looking to have more of a play (e.g. write extensions, use other AI and ML related Python components, such as tensorflow, mediapipe, transformers, StableDiffusion, pycuda, etc.)

## Python

SCALE is written in Python, so you'll need to have that installed. Most Linux distros will come with this by default. If not, you'll need to `sudo apt -y install python3` or similar.

For Windows, you should download and install a copy of [Python 3.xx for Windows](https://www.python.org/downloads/windows/). I would typically go for a stable release that's a couple of versions older than the latest one, e.g. if the latest is 3.14.x, then I'd find and install 3.12.x. This is because a lot of the (huge and complex) Python suites that relate to AI and ML are quite slow-moving and have a lot of dependencies. If you're on the bleeding edge, you're usually ahead of what all of those support.

Make sure you tick any options to add Python to your PATH during the installation procedure!

## Create and Initialise a Python venv

Assuming you've created a folder called 'scale' and put all of these files into it, you should create a 'virtual environment', which is just a local folder containing a copy of the Python interpreter (and other bits) and is where all of the installed packages will go. This avoids installing everything globally and potentially clashing with whatever you already have installed. 

```bash
cd scale
python -m venv .llm-venv

# if you're running in bash on Windows
. .llm-venv/Scripts/activate      
# if you're running in PowerShell
.\.venv\Scripts\Activate.ps1

# Update the venv    
python -m pip install --upgrade pip wheel setuptools
```

## Install Tree-sitter (language parser framework)

I found that I needed to version-pin these modules, because different versions of them seem to have made breaking changes to their internal ABI, so they won't talk to each other unless they are all the same version!

```bash
pip install -U "tree-sitter==0.21.0" "tree-sitter-c==0.21.0" "tree-sitter-javascript==0.21.0"
```

## Install the CUDA-aware llama-cpp (large language models)

`llama-cpp` is the mechanism through which SCALE loads and interacts with a Large Language Model. The precise version of 'llama-cpp' depends upon the host system configuration (OS and CPU architecture), Python version, and which GPU you have (if any).

Assuming you have a recent NVIDIA GPU, you should run this command to figure out your CUDA version:
    `nvidia-smi`

In my case, it's 12.4.

If you don't have a suitable GPU, there are other flavours of llama-cpp-python that you can install, including a CPU-only version (which will probably be very slow!).

In my example, where I have Windows 11 64-bit, Python 3.11.9, and NVIDIA CUDA 12.4, I chose to install a pre-build binary 'wheel' of `llama-cpp` and its Python bindings:

```bash
pip install --no-cache-dir --force-reinstall https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp311-cp311-win_amd64.whl
```

You can have a look around pages like [this](https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/) to see which wheels exist and hopefully find one that works on your system. 

## Download a Large Language Model...

You'll likely need to sign up to [HuggingFace](ttps://huggingface.co/models?library=gguf&sort=trending) and there are various models require you to request access, which is straightforward and usually pretty quick. `llama-cpp` requires local (downloaded) models to use the `GGUF` format, so you need to find a model that includes files of that type. If it has just "\*.safetensors" files it won't work.

I have tried various models, notably:

- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF?show_file_info=qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf)
- [bartowski/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF?show_file_info=Meta-Llama-3.1-8B-Instruct-Q6_K.gguf)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login ...your login details...
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-q4_k_m*.gguf" --local-dir models/Qwen2.5-7B-Instruct-GGUF --local-dir-use-symlinks False
```

# GPU Tier List

Here's a slightly silly tier list of relative performance of GPU rigs for this sort of processing...

* **S**: H200, H100, A100, B200; multi-GPU rigs. Frontier training and huge local inference. Not consumer.
* **A**: RTX 5090 (32 GB). Top-end consumer. 20B+ local models comfortably, very long contexts, high throughput.
* **B**: RTX 4090 (24 GB), 4080/4070 Ti/Super, 3090. 13B–20B workable, long contexts at good speed.
* **C**: RTX 4070 desktop, 4060 desktop. 7–13B smooth in Q4–Q5, modest long-context.
* **D**: RTX 4060 Laptop 8 GB (my machine), 4050 Laptop 6 GB. 7–8B decent in Q4, 13B only with compromises.
* **E**: RTX 3050/1650 class laptops, older Quadros. 7B OK, anything larger is painful.
* **F**: CPU-only or iGPU. Proof-of-concept only (awful!).

## Licence

Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
