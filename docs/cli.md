# CLI: flags and worked examples

Every supported invocation pattern. See [CLAUDE.md](../CLAUDE.md) for the dev-environment paths (venv interpreter,
sibling model directory) that every command here assumes.

```bash
# Annotate a Python file in place (verbose). -c enables definition docstrings/header comments.
python scale.py -m /path/model.gguf -c path/to/file.py -o path/to/file.py -v

# Common runtime knobs used in practice (see tests.sh):
python scale.py -c file.py -o out.py -v --n-ctx 12288 --n-batch 256

# The block pass (Python, C, JS) splits bodies into paragraphs and optionally comments them; combinable with -c.
#   --block-spacing         : paragraph spacing only (blank-line breaks), no block comments.
#   --block-comments {low,medium,high} : also write block comments (implies the pass). high = keep all; medium = drop
#       bare restatements; low = only intent/gotcha notes. (-b is shorthand for --block-spacing.)
# --block-comment-style {line,block} picks // (default) or /* */ for C/JS block-pass comments (ignored for Python).
python scale.py -c --block-comments medium file.py -o out.py -v

# --line-length N wraps each inserted block comment to fit N columns (indent + comment prefix included), continuing
# on fresh comment lines; identifiers are never split. Default 0 = unwrapped. Online: a value passed to the emit is
# stored in the manifest and honoured at apply; a --line-length on the apply command overrides it.
python scale.py -c --block-comments medium --line-length 99 file.py -o out.py -v

# --overwrite-comments (online emit only): by default a substantive (multi-line) hand-written block comment is
# preserved - its chunk is pre-answered NONE so the stronger model leaves it untouched. Pass this to offer those
# comments up for rewriting instead. Existing single-line comments are always surfaced for the model to judge.
python scale.py --block-comments medium -l python --online --emit-manifest m.json --overwrite-comments "src/*.py"

# --file-doc adds/updates the top-of-file header description (Python module docstring, or C/JS header comment),
# preserving shebang/copyright/license byte-for-byte; combinable with -c and the block flags, and runs LAST.
python scale.py --file-doc -l c -m /path/model.gguf file.c -o out.c -v

# --project-doc gives every pass a short, cached "project blurb" (so descriptions know the file's place in the wider
# project). Auto-detects CLAUDE.md/README.* near the source; pass a path, or 'none' to disable.
python scale.py -c --block-comments medium --project-doc CLAUDE.md file.c -o out.c -v

# Targets accept files/dirs/globs (multiple targets are annotated IN PLACE; -o is single-target only). --reference
# adds read-only files (e.g. the project's headers) that are parsed into the call graph - they resolve calls and
# supply per-routine callee contracts to the targets that use them (they are never summarised whole, never edited).
python scale.py -c --block-comments medium -l c "src/**/*.c" --reference include/ -m /path/model.gguf -v

# --doc-site (C only) controls where an extern function is documented when headers+sources are annotated together:
# 'auto' (default) documents the target HEADER prototype (prose generated from the definition's body) and skips the
# .c's docstring; 'impl' keeps the legacy impl docstring (target prototypes are still documented, from the prototype).
python scale.py -c --block-comments medium -l c "src/*.c" "include/*.h" --doc-site auto -m /path/model.gguf -v

# The ONLINE mode (Python, C, JS): defer EVERY routine's comments to a stronger model via ONE run-level manifest.
# The emit is model-free and instant - the GGUF is never loaded - and the targets are left byte-for-byte untouched.
# The /scale skill drives the whole loop with Claude as the stronger model. (--offline is the default mode; the two
# flags are mutually exclusive, and --online requires --emit-manifest.)
python scale.py -c --block-comments medium -l python --online \
  --emit-manifest m.json "src/*.py"                      # emit: model-free; every def + block recipe requested
python scale.py --check-manifest m.json                  # model-free completeness counter (exit 1 while unfilled)
python scale.py --next-fragment m.json --fragment-size 8 # check out the next <=8 unfilled requests as a small
                                                         # self-contained fragment (m.frag-001.json; path printed).
                                                         # Repeated calls hand out disjoint batches, so filling
                                                         # agents run in PARALLEL; exit 1 when nothing remains.
python scale.py -l python --apply-manifest m.json "src/*.py"   # apply: no model loaded. Sibling m.frag-*.json are
                                                         # merged in first (first write wins; spent files deleted);
                                                         # unfilled slots error out and return to the pile.

# The online file-description round (runs AFTER --apply-manifest, both phases model-free): --emit-filedoc records
# each target's current skeleton + role + header-zone lines; the stronger model fills each file's range+description
# answer pair; --apply-filedoc splices through the same license veto and preservation guard as --file-doc.
python scale.py -l python --emit-filedoc fd.json "src/*.py"
python scale.py -l python --apply-filedoc fd.json "src/*.py"

# Verification (the local quality floor) is ON by default for the offline def/block passes: the deterministic
# backtick-grounding gate plus clean-context challenge turns (grounding / obviousness / story). --no-verify disables.

# The header-reword manifest (--emit-reword, requires the offline --file-doc) records every freshly spliced file
# description for a stronger model to reword with cross-file consistency; --apply-reword re-splices the answers
# model-free (each draft located by exact match through the preservation guard; a miss is a safe no-op).
# --check-manifest counts all three manifest kinds (scale / scale-filedoc / scale-reword).
python scale.py --file-doc -l c -m /path/model.gguf --emit-reword r.json "src/*.c"
python scale.py -l c --apply-reword r.json "src/*.c"

# --config-dir DIR overlays a project's own prompt-template overrides on top of the built-in scale-cfg, per file (so
# a repo pins its house style without editing the shared install; an override dir need only carry the templates it
# changes). When omitted, a scale-cfg/ or .scale-cfg/ found at or above the working directory is used automatically.
python scale.py -c --block-comments medium --config-dir ./my-scale-cfg file.py -o out.py -v

# Without -o, output is printed to stdout. --help lists all flags.
```

Notes:

- `tests/run_all.py` is the fast, model-free regression suite (see [tests.md](tests.md)). `tests/block_eval/` holds
  the model-dependent block-pass eval harnesses.
- `tests.sh` is an older end-to-end harness: it runs the tool against JS/C/Python sample files (paths are
  machine-specific to the author — adjust before running).
- `tune_n_batch.py` probes throughput to pick a good `--n-batch` for a given model/GPU.
- The default model path (`DEFAULT_MODEL` in `scale.py`) is a local Qwen2.5.1-Coder-7B-Instruct GGUF (Q5_K_M);
  override with `-m` (required in this dev environment — see CLAUDE.md).
