# SCALE — Source Code Annotation with LLM Engine

SCALE is a local, scriptable tool that adds or updates code comments and docstrings using an on-device LLM loaded via `llama.cpp`. It is designed to slot into existing build or CI pipelines and to operate deterministically with tight output control. 

![Illustration](./scale.png)

## What it does

SCALE will ingest a source file, parse it to find routines (functions, methods, classes) and will create one comment per routine, following the comment style template
(as defined in "scale-cfg"). The output will include these comments but aims to avoid touching the code itself, thus there should be no functional/behaviour change to
the program, only comments are touched.

You are encouraged to look at the templates in "scale-cfg" and to adjust them to suit your needs (style guides, etc.) and codebase. The more the examples given in these
files align with the code you are likely to process, the more effective SCALE will be in useful comment generation. There is one template per language.

The "scale-cfg/comment.txt" file is the main 'system prompt' which SCALE passes into the LLM to guide its behaviour. You may also want to edit this, but in general this
shouldn't be necessary.

## How it works

1. **CLI entry** parses arguments such as model path, context size, sampling parameters, and the requested operation (e.g. `--comment`). Defaults are tuned for small local models. 
2. **Model wrapper** (`LocalChatModel`) loads the GGUF, normalises chat formatting to the target family (Qwen, Llama-3, Llama-2, Mistral, Phi-3), and exposes non-streaming or streaming completions with consistent stop-tokens. It can also download models from the Hub with pattern filters. 
3. **Priming** reads the global comment prompt and a language template from `scale-cfg`, confirms ingestion, and asks the model to summarise the full source once. The summary is then used as compact context for subsequent chunked requests. 
4. **Language worker**:
   * **Python**: parses the file into an AST, discovers definitions with precise spans, assembles self-contained snippets (headers plus bodies with child definitions replaced by stubs), asks the LLM for a docstring per definition, then applies patches back into the source while preserving formatting and comments. 
   * **Other languages**: the driver is plumbed to call language modules (e.g. JavaScript or C). 

Using Python as an example, SCALE will find every "class", "def", and "async def" in the input file and process them from the most deeply-nested up. For example:

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

The commenting process will start by ingesting the whole source file and asking the LLM to create a summary description of it. This summary is provided to the LLM
for each subsequent comment-generation turn.

Next, SCALE will extract the `mad()` function - its signature, any preceding decorators, and the body code - and write the docstring for it. If there was already
a docstring following the signature, that will be updated/replaced.

Next, SCALE will move to `bah()`. It will keep the signature of `mad()` but just include the (new) docstring. The body code for `mad()` is discarded, as it is a
potential distraction.

Once we have the docstring for `bah()`, we can move on to `__init__()`. Again, the body of `bah()` is replaced with just its docstring here and `mad()` is removed
altogether. Thus, SCALE is only looking at the code directly within the scope of `__init__()` and the signatures and docstrings for any functions that are
immediately nested within (and not any more deeply nested ones).

Finally, it reaches class `Foo`. It will look at the class signature (and any preceding decorators) and the signatures and docstrings of the methods - in this case,
just `__init__()`.

Once all the docstrings have been generated, the original source code is patched to place them into the correct locations, replacing any pre-existing docstrings as
required. This patching process ensures that the executable code, indentation, other comments and empty lines are all preserved without any risk of the LLM
inadvertently editing them or hallucinating more code!

## Key design choices

* **Local-first**: runs entirely against GGUF models; no cloud dependency. 
* **Deterministic framing**: strict chat templates and stop strings per model family reduce role-token leaks and over-generation. 
* **Safe context management**: estimates bytes-per-token on the fly and warns before prompts approach the context window. 

## Using SCALE (typical)

```bash
# Generate comments/docstrings for a source file with a local GGUF model
python scale.py --verbose --comment -m /path/to/model.gguf /path/to/file.py -o /path/to/output.py
```

Use `--help` for more guidance on the CLI interface.

You can override chat format detection (`--format`), temperature/top-p/top-k, repeat penalty, context size, batch size, and GPU offload layers to suit different cards and quantisations. 

## Extensibility

* Add new languages by implementing a `generate_language_comments` function matching the Python worker’s interface and wiring it in the dispatcher. 
* Extend or swap prompt templates by editing files under `scale-cfg` without changing code. 

## Current scope and limits

* Quality and speed depend on the chosen GGUF quant, context size and GPU memory. The wrapper exposes these controls but does not auto-tune them. 
* The current implementation only supports Python, C and JavaScript source code.
* This has been tested on a Windows 11 laptop with an NVIDIA RTX 4060 mobile GPU. Better hardware than this will be able to utilise stronger LLMs.

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

# GPU Tier List

Here's a slightly silly tier list of relative performance of GPU rigs for this sort of processing...

* **S**: H200, H100, A100, B200; multi-GPU rigs. Frontier training and huge local inference. Not consumer.
* **A**: RTX 5090 (32 GB). Top-end consumer. 20B+ local models comfortably, very long contexts, high throughput.
* **B**: RTX 4090 (24 GB), 4080/4070 Ti/Super, 3090. 13B–20B workable, long contexts at good speed.
* **C**: RTX 4070 desktop, 4060 desktop. 7–13B smooth in Q4–Q5, modest long-context.
* **D**: RTX 4060 Laptop 8 GB (my machine), 4050 Laptop 6 GB. 7–8B decent in Q4, 13B only with compromises.
* **E**: RTX 3050/1650 class laptops, older Quadros. 7B OK, anything larger is painful.
* **F**: CPU-only or iGPU. Proof-of-concept only (awful!).
