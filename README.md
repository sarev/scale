# SCALE — Source Code Annotation with LLM Engine

SCALE is a local, scriptable tool that adds or updates code comments and docstrings using an on-device LLM loaded
via `llama.cpp`. It is designed to slot into existing build or CI pipelines and to operate deterministically with
tight output control.

While SCALE can take a lot of the pain out of creating and maintaining comments on a codebase, you should still
review its output and potentially make an editorial pass - just like you would in a code review of a team mate's
work.

![Illustration](./scale.png)

## What it does

SCALE will ingest a source file, parse it to find routines (functions, methods, classes) and will create one
comment per routine, following the comment style template (as defined in "scale-cfg"). The output will include
these comments but aims to avoid touching the code itself, thus there should be no functional/behaviour change
to the program, only comments are touched.

You are encouraged to look at the templates in "scale-cfg" and to adjust them to suit your needs (style guides,
etc.) and your specific codebase. The more the examples given in these files align with the code you are likely
to process, the more effective SCALE will be in useful comment generation. There is one template per supported
language.

The "scale-cfg/comment.txt" file is the main 'system prompt' which SCALE passes into the LLM to guide its
behaviour. You may also want to edit this, but in general this
shouldn't be necessary.

## How it works

1. **CLI entry** parses arguments such as model path, context size, sampling parameters, and the requested
   operation (e.g. `--comment`). Defaults are tuned for small local models. 
2. **Model wrapper** (`LocalChatModel`) loads the GGUF, normalises chat formatting to the target family (Qwen,
   Llama-3, Llama-2, Mistral, Phi-3), and exposes non-streaming or streaming completions with consistent stop-tokens.
   It can also download models from the Hub with pattern filters. 
3. **Priming** reads the global comment prompt and a language template from `scale-cfg`, confirms ingestion, and
   asks the model to summarise the full source once. The summary is then used as compact context for subsequent
   chunked requests. 
4. **Language worker**: parses the input file and adds/replaces comments on functions, classes, methods with
   versions that align to the templates supplied in the configuration. Currently, there are three language
   workers: Python, C, and JavaScript.

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

The commenting process will start by ingesting the whole source file and asking the LLM to create a summary
description of it. This summary is provided to the LLM for each subsequent comment-generation turn. It is
useful to give SCALE some understanding of the whole file, because it helps with comprehension of each function,
class, and method as it is processing, but we just use the summary to help keep the context window small (i.e.
to help performance).

The 'conversation' that is maintained during this process is between the 'user' (the tool) and the 'assistant'
(the LLM). On each turn, the user asks the assistant to generate a comment for the block of code they have
supplied (a function, method, or class). These turns are not recorded, so the LLM isn't building an
ever-increasing context window - we discard each chunk once it's done. All the LLM has to go on is:

1. The system prompt
2. The summary of the program as a whole
3. The comment template prompt
4. The function/method/class it's been asked to generate a comment for

In the Python example above, SCALE will extract the `mad()` function - its signature, any preceding decorators,
and the body code - and write the docstring for it. If there was already a docstring following the signature,
that will be updated/replaced.

Next, SCALE will move to `bah()`. It will keep the signature of `mad()` but just include the (new) docstring.
The body code for `mad()` is discarded, as it is a potential distraction. In a way, the LLM is 'remembering'
the nested function `mad()` because it has been given the docstring (it generated earlier) for it.

Once we have the docstring for `bah()`, we can move on to `__init__()`. Again, the body of `bah()` is replaced
with just its docstring here and `mad()` is removed altogether. Thus, SCALE is only looking at the code directly
within the scope of `__init__()` and the signatures and docstrings for any functions that are immediately nested
within (and not any more deeply nested ones).

Finally, it reaches class `Foo`. It will look at the class signature (and any preceding decorators) and the
signatures and docstrings of the methods - in this case, just `__init__()`. So the comment generated for the
class will be based upon a collective assessment of what all the method docstrings say, plus whatever class
attributes there might be (as well as any pre-existing docstring the class may have had).

Once all the docstrings have been generated, the original source code is patched to place them into the correct
locations, replacing any pre-existing docstrings as required. This patching process ensures that the executable
code, indentation, other comments and empty lines are all preserved without any risk of the LLM inadvertently
editing them or hallucinating more code!

## Key design choices

* **Local-first**: runs entirely against GGUF models; no cloud dependency. 
* **Deterministic framing**: strict chat templates and stop strings per model family reduce role-token leaks and
  over-generation. 
* **Safe context management**: estimates bytes-per-token on the fly and warns before prompts approach the context
  window. 

## Using SCALE (typical)

```bash
# Generate comments/docstrings for a source file with a local GGUF model
python scale.py --verbose --comment -m /path/to/model.gguf /path/to/file.py -o /path/to/output.py
```

Use `--help` for more guidance on the CLI interface.

You can override LLM features, such as the chat format detection (`--format`), temperature/top-p/top-k, repeat
penalty, context size, batch size, and GPU offload layers to suit different models, hardware and quantisations.

## Extensibility

* Add new languages by implementing a `generate_language_comments` function matching the Python worker’s interface
  and wiring it in the dispatcher. 
* Extend or swap prompt templates by editing files under `scale-cfg` without changing code. 

## Current scope and limits

* The current implementation only supports Python, C and JavaScript source code.
* Quality and speed depend on the chosen model, GGUF quant, context size and GPU memory. The wrapper exposes
  these controls but does not auto-tune them to your system.
* This has been tested on a Windows 11 laptop with an NVIDIA RTX 4060 mobile GPU. Better hardware than this will
  be able to utilise stronger LLMs.

# Installation

I've included a lot of optional steps below. Skip over them if you just want to try SCALE out - these are only
really useful for people who are looking to have more of a play (e.g. write extensions, use other AI and ML
related Python components, such as tensorflow, mediapipe, transformers, StableDiffusion, pycuda, etc.)

## Python

SCALE is written in Python, so you'll need to have that installed. Most Linux distros will come with this by
default. If not, you'll need to `sudo apt -y install python3` or similar.

For Windows, you should download and install a copy of [Python 3.xx for Windows](https://www.python.org/downloads/windows/).
I would typically go for a stable release that's a couple of versions older than the latest one, e.g. if the
latest is 3.14.x, then I'd find and install 3.12.x. This is because a lot of the (huge and complex) Python
suites that relate to AI and ML are quite slow-moving and have a lot of dependencies. If you're on the
bleeding edge, you're usually ahead of what all of those support.

Make sure you tick any options to add Python to your PATH during the installation procedure!

## Optional - NVIDIA CUDA Toolkit and cuDNN

I don't *think* you need to do any of this, but if you run into issues getting the later pip installs to work or
you want to do more experimentation with AI and machine learning, then these are very useful:

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

## Optional - Visual Studio C/C++ Compiler

This is only required if you need to compile CUDA kernels or build Python wheels, etc. You shouldn't need this
just to run SCALE, but I include it for completeness for those who are looking to do AI-related development
on a Windows system.

- Install Visual Studio C and C++ support...
  - Pick the "Desktop development with C++ workload" in the installer
  - [Install C and C++ support in Visual Studio](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)

## Create and initialise a Python venv

Assuming you've created a folder called 'scale' and put all of these files into it...

```bash
cd scale
python -m venv .llm-venv
. .llm-venv/Scripts/activate
python -m pip install --upgrade pip wheel setuptools
```

## Optional - run the CUDA activation script

Don't do this unless you are planning on doing more sophisticated things than just running SCALE...

This program attempts to locate important parts of typical AI software infrastructure on your system and
updates PATH in the local shell to ensure that the right things are present. It should work on Linux and
Windows (including in bash on Windows - e.g. Git Bash for Windows).

It only affects the terminal session that it is run within.

### Optional - Running `cuda_activate.py` bash in Windows

```bash
eval "$(python cuda_activate.py --shell bash)"
```

### Optional - Running `cuda_activate.py` in Windows PowerShell

```powershell
python cuda_activate.py --shell powershell | Invoke-Expression
```

### Optional - Running `cuda_activate.py` in Windows cmd.exe

```batch
for /f "usebackq delims=" %i in (`python cuda_activate.py --shell cmd`) do %i
```

This script is probably a little fragile and may require some tweaks for your setup.

## Install the CUDA-aware llama-cpp (large language models)

Precise version depends upon the host system:

    `nvidia-smi`
 
should tell you the CUDA version you have. In my case, it's 12.4...

If you don't have a suitable GPU, there are other flavours of llama-cpp-python that you can install,
including a CPU-only version (which will probably be very slow!).

```bash
pip install --no-cache-dir --force-reinstall https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp311-cp311-win_amd64.whl
```

## Tree-sitter (language parser framework)

```bash
pip install -U "tree-sitter==0.21.0" "tree-sitter-c==0.21.0" "tree-sitter-javascript==0.21.0"
```

## Download a Large Language Model...

I have tried various models from <https://huggingface.co>, notably:

- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF?show_file_info=qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf)
- [bartowski/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF?show_file_info=Meta-Llama-3.1-8B-Instruct-Q6_K.gguf)

You'll likely need to sign up to HuggingFace and various models require you to request access, which is
straightforward and usually pretty quick.

```bash
pip install -U "huggingface_hub[cli]"
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
