# SCALE — Source Code Annotation with LLM Engine

SCALE writes and updates code comments using a local LLM. Point it at Python, C, or JavaScript source and it adds docstrings and header comments to every routine, paragraphs and comments the insides of long functions, and writes a top-of-file description — while guaranteeing that the code itself is untouched.

That guarantee is structural, not a promise about model behaviour: SCALE parses your source, asks the model only for comment text, and patches that text into the original file. The model never re-emits your code, so executable code, indentation, blank lines, and comments it wasn't asked to touch are preserved byte-for-byte. If a generated comment can't be placed safely, it is dropped rather than risked.

Everything runs on your own machine via `llama.cpp` and a GGUF model — no code leaves your system unless you opt into the Claude Code integration described below. You should still review the output, as you would a teammate's pull request, but you'll be editing prose, not untangling broken code.

![Illustration](./scale.png)

## What it writes

Each kind of comment is a separate pass; enable any combination in one run.

| Pass | Flag | What you get |
| --- | --- | --- |
| Definition docs | `-c` | A docstring (Python) or header comment (C/JS) for every function, method, and class |
| Block comments | `--block-comments low\|medium\|high` | Long function bodies split into logical paragraphs, with a comment on each group that earns one (`--block-spacing` for paragraphing only) |
| File description | `--file-doc` | A top-of-file summary of what the file is for — shebangs, copyright, and licence blocks preserved exactly |
| C doc-sites | `--doc-site auto` | When C headers and sources are annotated together, public functions are documented on their header prototype — where API users actually read |

Existing documentation is an input, not an obstacle: SCALE reads what's already there and updates it, so hand-written rationale ("we retry because the vendor API is flaky") is carried forward rather than flattened into a description of the code.

## Quality controls

A small local model writes plausible prose; left unchecked, some of it would be wrong or worthless. SCALE doesn't trust it:

- **Claims are verified against the code.** Every generated comment is challenged in a fresh model context — do the things it names actually appear in the source? Is it telling the reader something the code doesn't already say? Failures are regenerated once, then dropped or flagged in the run output.
- **Restatements are filtered.** A comment that just narrates the line below it ("increment the counter") is rejected. The density knob (`low`/`medium`/`high`) sets how strict this is.
- **Completion is counted, not assumed.** Multi-file runs track every routine; nothing is silently skipped.

The prompt templates live in `scale-cfg/` as plain text files. Edit them to match your house style — comment format, tone, what a good comment looks like in your codebase. No code changes needed.

## Project awareness

Annotating a file in isolation produces isolated-sounding docs. SCALE can do better:

- **Multi-file runs.** Targets accept files, directories, and globs; multiple targets are annotated in place in one model load.
- **Call-graph ordering.** Within a run, callees are documented before their callers, and each routine's prompt includes one-line contracts for the functions it calls — so a caller's docs can say *why* it delegates, not just that it does.
- **Read-only references.** `--reference include/` parses extra files (e.g. your headers) for context without ever editing them.
- **Project blurb.** `--project-doc` feeds every pass a short description of the wider project (auto-detected from a nearby `README`/`CLAUDE.md`), so file descriptions know their place in the system.

## Two modes: offline and online

**Offline is the default.** Every pass, including verification, runs entirely on your own hardware via the local GGUF model — no network, nothing leaves your system.

**Online** (`--online`) trades that privacy for quality: every routine's comments are written by a stronger model (in practice, Claude via Claude Code) instead of the local one. SCALE collects the work into a **manifest** — a single JSON file of self-contained units (each routine's code plus context, and empty slots for the answers). The emit is model-free and finishes in seconds (the GGUF is never loaded); the stronger model fills in the slots; SCALE applies the answers back through the same code-preserving patcher, and a completeness check counts what's still unfilled. The structure stays SCALE's: segmentation, placement, and the byte-for-byte guarantee are deterministic machinery, and the stronger model is only ever handed bounded, machine-checkable units — never an open-ended "comment this codebase".

```bash
python scale.py -c --block-comments medium -l c --online \
    --emit-manifest scale-manifest.json "src/*.c"      # model-free, instant
# ...a stronger model fills in the manifest answers...
python scale.py --check-manifest scale-manifest.json   # exits non-zero until complete
python scale.py -l c --apply-manifest scale-manifest.json "src/*.c"   # no model loaded
# ...then the file descriptions, as a second small manifest round...
python scale.py -l c --emit-filedoc scale-filedoc.json "src/*.c"
python scale.py -l c --apply-filedoc scale-filedoc.json "src/*.c"
```

The repository ships a Claude Code skill (`.claude/skills/scale/`) that drives this loop end-to-end: ask Claude Code to "scale" some files and it emits the manifest, fills it (fanning out parallel subagents over self-contained fragments), applies the answers, runs the file-description round, and verifies the counts. Claude's answers go through the same machinery as everything else.

**None of this is required.** Without `--online`, SCALE is a fully offline tool.

## Using SCALE

```bash
# Docstrings for every routine in a file, written to a new file
python scale.py -c src/parser.py -o out/parser.py -v

# In place, adding paragraphing and block comments inside long bodies
python scale.py -c --block-comments medium src/parser.py -o src/parser.py -v

# The works, for a C codebase: sources + headers together, API docs on the prototypes
python scale.py -c --block-comments medium --file-doc -l c \
    "src/*.c" "include/*.h" --doc-site auto -v
```

Useful to know:

- Multiple targets are annotated **in place** (commit first); a single target may use `-o`; with no `-o`, output goes to stdout.
- `-m /path/model.gguf` selects the model. The default path is `models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf` under the project root.
- `-l python|c|js` forces the language — recommended, since auto-detection reads the content, not the file extension.
- Runtime knobs (`--n-ctx`, `--n-batch`, GPU offload, sampling) are exposed for tuning to your hardware; `--help` lists everything, and [docs/cli.md](docs/cli.md) has a worked example for every flag.

Files larger than the model's context window are handled automatically (summarised in chunks; oversized bodies elided from the model's *view* only), and binary-safe I/O preserves your line endings and any unusual bytes exactly.

## Language support

| Language | Parser | Definition docs | Block comments | File description |
| --- | --- | --- | --- | --- |
| Python | CPython `ast` | docstrings | ✓ | module docstring |
| C | tree-sitter | header comments (+ `--doc-site`) | ✓ | header comment |
| JavaScript | tree-sitter (ESM + CommonJS) | JSDoc | ✓ | header comment |

Adding a language means implementing one well-defined worker interface; see [docs/architecture.md](docs/architecture.md).

## Tests

`tests/` contains a fast, self-contained regression suite — no model needed (the LLM is stubbed), so it runs in seconds:

```bash
python tests/run_all.py    # everything, with a pass/fail summary; non-zero exit on failure
```

## Scope and limits

- Python, C, and JavaScript only, for now.
- Offline output quality tracks the model you give it. A code-tuned 7B writes good summaries and serviceable walkthroughs; consistently trustworthy, insightful commentary is what the online mode is for.
- SCALE cannot invent institutional knowledge — design rationale and cross-system context that only live in someone's head. Where that knowledge is already written down, it is preserved and updated; where it isn't, you still need a human.

## Picking an LLM

Models in GGUF format are downloaded from [HuggingFace](https://huggingface.co/models?library=gguf&sort=trending). The default — and the best of those tested for comment quality — is [bartowski/Qwen2.5.1-Coder-7B-Instruct](https://huggingface.co/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF), a code-specialised 7B whose Q5_K_M quant is ~5.2 GB.

Rule of thumb for sizing: a model needs roughly *parameters × bytes-per-parameter* of VRAM, plus 25–30% headroom for the context window. Quantisation shrinks bytes-per-parameter (Q4 ≈ half a byte) at a small cost in smarts. On an 8 GB GPU, a 7–8B model at Q5–Q6 with a ~12k context is the comfortable ceiling; spill past your VRAM and performance falls off a cliff.

### GPU tier list

A slightly silly tier list for this sort of processing:

* **S**: H200, H100, A100, B200; multi-GPU rigs. Frontier training and huge local inference. Not consumer.
* **A**: RTX 5090 (32 GB). Top-end consumer. 20B+ local models comfortably, very long contexts, high throughput.
* **B**: RTX 4090 (24 GB), 4080/4070 Ti/Super, 3090. 13B–20B workable, long contexts at good speed.
* **C**: RTX 4070 desktop, 4060 desktop. 7–13B smooth in Q4–Q5, modest long-context.
* **D**: RTX 4060 Laptop 8 GB, 4050 Laptop 6 GB. 7–8B decent in Q4, 13B only with compromises.
* **E**: RTX 3050/1650 class laptops, older Quadros. 7B OK, anything larger is painful.
* **F**: CPU-only or iGPU. Proof-of-concept only (awful!).

## Installation — Linux

Assumes a Debian-based system with an NVIDIA GPU and nothing installed yet; skip the steps you already have.

```bash
# Python and build tools
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential git cmake pkg-config

# NVIDIA driver (skip without a GPU); needs contrib/non-free-firmware enabled
sudo sed -i 's/main$/main contrib non-free-firmware/' /etc/apt/sources.list
sudo apt update
sudo apt install -y nvidia-driver
sudo reboot

# Verify the driver and note your CUDA version
nvidia-smi

# Get SCALE and create a virtual environment
git clone https://github.com/sarev/scale.git
cd scale
python3 -m venv env
. env/bin/activate
python -m pip install -U pip setuptools wheel

# Parsers. The tree-sitter runtime and grammars share an ABI - install all three together.
pip install -U tree-sitter tree-sitter-c tree-sitter-javascript

# llama.cpp bindings - pick ONE of these:
# (a) CUDA wheel (cuXXX = your CUDA version from nvidia-smi)
pip install -U llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
# (b) build from source with CUDA
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install -U llama-cpp-python
# (c) CPU-only (very slow!)
pip install -U llama-cpp-python

# Download the default model (~5.2 GB) to where SCALE looks for it
pip install -U "huggingface_hub[cli]"
hf download bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF --include "*Q5_K_M.gguf" \
    --local-dir models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF

# Annotate something
python scale.py -c path/to/input.py -o out/annotated.py -v
```

## Installation — Windows

Install [Python 3.x for Windows](https://www.python.org/downloads/windows/) — a stable release a version or two behind the latest is the safe choice, since prebuilt ML wheels lag the newest Python. Tick the option to add Python to your PATH.

```bash
# From the SCALE checkout (bash syntax; for PowerShell activate with .\env\Scripts\Activate.ps1)
python -m venv env
. env/Scripts/activate
python -m pip install --upgrade pip wheel setuptools

# Parsers. The tree-sitter runtime and grammars share an ABI - install all three together.
pip install -U tree-sitter tree-sitter-c tree-sitter-javascript
```

For the llama.cpp bindings, run `nvidia-smi` to find your CUDA version, then install a matching prebuilt wheel — browse the [release pages](https://github.com/abetlen/llama-cpp-python/releases) or the [wheel index](https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/) for your OS / Python / CUDA combination. For example (Windows 11 x64, Python 3.11, CUDA 12.4):

```bash
pip install --no-cache-dir --force-reinstall https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp311-cp311-win_amd64.whl
```

CPU-only builds exist too (`pip install llama-cpp-python`) but will be painfully slow.

Finally, download the default model to where SCALE looks for it:

```bash
pip install -U "huggingface_hub[cli]"
hf download bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF --include "*Q5_K_M.gguf" \
    --local-dir models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF
```

(`hf` is the current HuggingFace CLI; `huggingface-cli` still works as a deprecated alias. Some models on HuggingFace require a quick access request and `hf auth login` first.)

Sanity-check the install with the model-free test suite, then a real run:

```bash
python tests/run_all.py
python scale.py -c path/to/input.py -o out/annotated.py -v
```

## Licence

Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
