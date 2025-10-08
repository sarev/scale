#!/usr/bin/env python3
"""
Tune n_batch for a llama.cpp model by probing throughput on a fixed prompt.

It reloads the model for each candidate n_batch, measures tokens/sec on a short
completion, and stops when throughput drops beyond a tolerance or an error occurs
(e.g. CUDA OOM). The best n_batch and a summary table are printed.

Examples
--------
Basic:
  python tune_n_batch.py --model /path/model.gguf --prompt "Hello world" --n-ctx 8192

With messages JSON (OpenAI-like chat format):
  python tune_n_batch.py --model /path/model.gguf --messages messages.json --chat-format auto

Power users (bigger sweep):
  python tune_n_batch.py --model /path/model.gguf --prompt-file prompt.txt --start 64 --max-n 4096 --growth 1.5

Export results to JSON:
  python tune_n_batch.py --model /path/model.gguf --prompt "..." --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional: reuse your project’s formatters if available
_FORMATTERS: Dict[str, Any] = {}
try:
    # scale_llm.py should define FORMATTERS and compatible message building
    from scale_llm import FORMATTERS as SCALE_FORMATTERS  # type: ignore
    _FORMATTERS = dict(SCALE_FORMATTERS)
except Exception:
    _FORMATTERS = {}


# Minimal fallback chat formatter if scale_llm is not available
def _fallback_formatter(messages: List[Dict[str, str]]) -> Tuple[str, List[str]]:
    """
    Build a plain prompt by concatenating messages with role tags.

    This function takes a list of message dictionaries and returns a tuple containing the constructed prompt and
    an empty list of stop tokens.

    Parameters:
    - `messages`: A list of dictionaries, where each dictionary contains a message with optional 'role' and 'content' keys.

    Returns:
    - A tuple containing the constructed plain prompt as a string and an empty list of stop tokens.
    """
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"[{role}]\n{content}\n")
    prompt = "\n".join(parts) + "[assistant]\n"
    return prompt, []


# ---------------- probe logic ----------------


@dataclass
class BatchProbeResult:
    """
    Represents the result of a batch probe operation.

    This class encapsulates the outcome of probing different `n_batch` values for a given model.
    It provides metrics for evaluation throughput, prompt throughput, and wall time, as well as an indication of
    whether the operation was successful.

    Parameters:
    - `n_batch`: The batch size value probed.
    - `ok`: A boolean indicating whether the probe operation was successful.
    - `eval_tps`: The evaluation tokens per second metric.
    - `eval_tokens`: The total number of evaluation tokens processed.
    - `prompt_tps`: The prompt tokens per second metric (0.0 if unavailable).
    - `total_ms`: The total time taken for the probe operation in milliseconds.
    - `wall_tps`: The completion tokens per wall time metric (fallback).
    - `error`: An optional error message if the probe operation failed.
    """

    n_batch: int
    ok: bool
    eval_tps: float          # eval tokens/sec
    eval_tokens: int
    prompt_tps: float        # prompt tokens/sec (0.0 if unavailable)
    total_ms: float
    wall_tps: float          # completion tokens / wall time (fallback)
    error: Optional[str] = None


def _build_prompt(
    prompt: Optional[str],
    prompt_file: Optional[Path],
    messages_file: Optional[Path],
    chat_format: str
) -> Tuple[str, List[str]]:
    """
    Construct a fixed prompt and stop tokens.

    This function builds a prompt from various sources, prioritizing the following order:

    1. Messages file: If provided, loads a JSON list of messages from the file and formats them according
       to the specified chat format.
    2. Prompt file: If no messages file is provided, attempts to read a prompt from the specified file.
    3. Default prompt: If neither a messages file nor a prompt file is provided, returns a default prompt.

    Parameters:
    - `prompt`: The prompt text as a string (optional).
    - `prompt_file`: The path to a file containing the prompt text (optional).
    - `messages_file`: The path to a JSON file containing a list of messages (optional).
    - `chat_format`: The format for rendering the messages (e.g., "json", "markdown").

    Returns:
    - A tuple containing the constructed prompt text and a list of stop tokens.
    """
    if messages_file:
        with open(messages_file, "r", encoding="utf-8") as f:
            msgs = json.load(f)
            if not isinstance(msgs, list):
                raise ValueError("--messages must be a JSON list of {role, content} dicts")
        fmt = _FORMATTERS.get(chat_format.lower()) if _FORMATTERS else None
        if fmt is None:
            return _fallback_formatter(msgs)
        prompt_text, stops = fmt(msgs)
        return prompt_text, list(stops or [])
    if prompt_file:
        text = Path(prompt_file).read_text(encoding="utf-8")
        return text, []
    if prompt is not None:
        return prompt, []
    # last resort
    return "You are a concise assistant.\nUser: Say hello.\nAssistant:", []


def _measure_once(
    *,
    model_path: str,
    prompt: str,
    stops: List[str],
    n_ctx: int,
    n_batch: int,
    n_gpu_layers: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    seed: Optional[int],
    threads: Optional[int],
    verbose_llama: bool,
    **llama_kwargs: Any
) -> BatchProbeResult:
    """
    Measure tokens/sec on a short completion for a given n_batch value.

    This function loads a temporary llama.cpp model with the given n_batch and measures tokens/sec
    on a short completion. It uses llama.cpp timings if available, falling back to usage + wall clock
    if timings are missing.

    Parameters:
    - `model_path`: The path to the llama.cpp model.
    - `prompt`: The prompt text for the completion.
    - `stops`: A list of stop words for the completion.
    - `n_ctx`: The context length for the completion.
    - `n_batch`: The batch size for the completion.
    - `n_gpu_layers`: The number of GPU layers for the completion.
    - `max_new_tokens`: The maximum number of new tokens for the completion.
    - `temperature`: The temperature for the completion.
    - `top_p`: The top-p value for the completion.
    - `top_k`: The top-k value for the completion.
    - `repeat_penalty`: The repeat penalty for the completion.
    - `seed`: The random seed for the completion (optional).
    - `threads`: The number of threads for the completion (optional).
    - `verbose_llama`: Whether to enable verbose logging for llama.cpp (optional).
    - `**llama_kwargs`: Additional keyword arguments for llama.cpp.

    Returns:
    - A BatchProbeResult object containing the measured tokens/sec, eval count, prompt eval count,
      wall time, and any error message.
    """
    try:
        from llama_cpp import Llama  # import here to keep script import-light
    except Exception as e:
        raise SystemExit(f"ERROR: llama_cpp is required: pip install llama-cpp-python\n{e}")

    t0 = time.perf_counter()
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            seed=seed if seed is not None else 0,
            n_threads=threads if threads is not None else None,
            verbose=verbose_llama,
            **llama_kwargs
        )
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        return BatchProbeResult(n_batch, False, 0.0, 0, 0.0, (time.perf_counter() - t0) * 1000.0, error=str(e))

    try:
        t1 = time.perf_counter()
        resp: Dict[str, Any] = llm.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stops or None
        )
        t2 = time.perf_counter()

        timings = resp.get("timings") or {}
        usage = resp.get("usage") or {}
        # Prefer llama.cpp timings, fall back to usage + wall clock
        eval_count = int(timings.get("eval_count") or usage.get("completion_tokens") or 0)
        eval_ms = float(timings.get("eval_ms") or 0.0)
        prompt_eval_count = int(timings.get("prompt_eval_count") or usage.get("prompt_tokens") or 0)
        prompt_ms = float(timings.get("prompt_ms") or 0.0)

        wall_s = max(1e-6, (t2 - t1))
        eval_tps = (eval_count / (eval_ms / 1000.0)) if eval_ms > 0 else (eval_count / wall_s if eval_count > 0 else 0.0)
        prompt_tps = (prompt_eval_count / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0.0
        wall_tps = (eval_count / wall_s) if eval_count > 0 else 0.0

        err = None
        if eval_count == 0:
            err = "no sampled tokens (timings missing or immediate stop)"
        return BatchProbeResult(n_batch, True, eval_tps, eval_count, prompt_tps, wall_tps, (t2 - t0) * 1000.0, err)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        return BatchProbeResult(n_batch, False, 0.0, 0, 0.0, (time.perf_counter() - t0) * 1000.0, error=str(e))


def probe_optimal_n_batch(
    *,
    model_path: str,
    prompt: str,
    stops: List[str],
    n_ctx: int,
    n_gpu_layers: int,
    start: int,
    max_n: int,
    growth: float,
    warmup_passes: int,
    drop_tolerance: float,
    sample_cfg: Dict[str, Any],
    verbose: bool,
    verbose_llama: bool,
    objective: str = "eval_tps",
    **llama_kwargs: Any
) -> Tuple[int, List[BatchProbeResult]]:
    """
    Increase n_batch geometrically and measure throughput until it drops by tolerance or an error occurs.

    This function probes the optimal `n_batch` value for a given model by measuring throughput on a fixed prompt.
    It reloads the model for each candidate `n_batch`, measures tokens/sec on a short completion, and stops when
    throughput drops beyond a tolerance or an error occurs.

    Parameters:
    - `model_path`: The path to the model file.
    - `prompt`: The fixed prompt text.
    - `stops`: A list of stop sequences.
    - `n_ctx`: The context length.
    - `n_gpu_layers`: The number of GPU layers.
    - `start`: The starting value for `n_batch`.
    - `max_n`: The maximum value for `n_batch`.
    - `growth`: The geometric growth factor.
    - `warmup_passes`: The number of warm-up passes.
    - `drop_tolerance`: The tolerance for dropping throughput.
    - `sample_cfg`: A dictionary of sample configuration parameters.
    - `verbose`: Whether to print verbose logging.
    - `verbose_llama`: Whether to print llama initialization logs.
    - `objective`: The optimization target (default: "eval_tps").
    - `llama_kwargs`: Additional keyword arguments for llama initialization.

    Returns:
    - A tuple containing the best `n_batch` value and a list of batch probe results.
    """
    # Warm up at a conservative setting to avoid cold-start bias
    if warmup_passes > 0:
        _ = _measure_once(
            model_path=model_path,
            prompt=prompt,
            stops=stops,
            n_ctx=n_ctx,
            n_batch=max(16, start),
            n_gpu_layers=n_gpu_layers,
            verbose_llama=False,
            **sample_cfg,
            **llama_kwargs
        )

    results: List[BatchProbeResult] = []
    best_val = -float("inf")
    best_n = start

    n = start
    while n <= max_n:
        r = _measure_once(
            model_path=model_path,
            prompt=prompt,
            stops=stops,
            n_ctx=n_ctx,
            n_batch=n,
            n_gpu_layers=n_gpu_layers,
            verbose_llama=verbose_llama,
            **sample_cfg,
            **llama_kwargs
        )
        results.append(r)

        if verbose:
            status = "OK " if r.ok else "ERR"
            print(
                f"[probe] n_batch={n:<5d} {status}  eval_tps={r.eval_tps:8.2f}  "
                f"prompt_tps={r.prompt_tps:8.2f}  wall_tps={r.wall_tps:8.2f}  "
                f"eval_tokens={r.eval_tokens:<5d}  ms={r.total_ms:8.1f}"
                + (f"  err={r.error}" if r.error else "")
            )

        if not r.ok:
            break

        # Choose metric
        if objective == "eval_tps":
            metric = r.eval_tps
        elif objective == "prompt_tps":
            metric = r.prompt_tps
        elif objective == "wall_tps":
            metric = r.wall_tps
        elif objective == "total_ms":
            metric = -r.total_ms  # lower is better
        else:
            metric = r.eval_tps

        if metric >= best_val:
            best_val = metric
            best_n = n
        else:
            # Stop when we drop by tolerance from best (only for "higher is better" metrics)
            if objective in ("eval_tps", "prompt_tps", "wall_tps") and best_val > 0:
                if (best_val - metric) / best_val >= drop_tolerance:
                    break

        n = int(max(n + 1, round(n * growth)))

    if verbose:
        pretty = {
            "eval_tps": f"eval_tps≈{max(best_val,0):.2f}",
            "prompt_tps": f"prompt_tps≈{max(best_val,0):.2f}",
            "wall_tps": f"wall_tps≈{max(best_val,0):.2f}",
            "total_ms": f"total_ms≈{-best_val:.1f} ms"
        }.get(objective, f"metric≈{best_val:.2f}")
        print(f"[probe] Best n_batch={best_n} with {pretty}")
    return best_n, results


# ---------------- CLI ----------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    """
    Parse command-line arguments for the probe and recommend llama.cpp n_batch for a fixed prompt.

    This function uses the `argparse` library to define and parse command-line arguments. The following options are supported:

    Parameters:
    - `argv`: List of command-line argument strings.

    Returns:
    - An `argparse.Namespace` object containing the parsed command-line arguments.
    """

    p = argparse.ArgumentParser(description="Probe and recommend llama.cpp n_batch for a fixed prompt.")
    p.add_argument("--model", required=True, help="Path to .gguf model.")
    g_in = p.add_mutually_exclusive_group()
    g_in.add_argument("--prompt", help="Inline prompt text.")
    g_in.add_argument("--prompt-file", type=Path, help="Path to a prompt text file.")
    g_in.add_argument("--messages", type=Path, help="JSON file of messages [{role, content}, ...].")
    p.add_argument("--chat-format", default="auto", help="Chat format key if using messages (honoured when scale_llm is available).")
    p.add_argument("--n-ctx", type=int, default=8192, help="Context length.")
    p.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of layers on GPU (-1 means auto or all supported).")
    p.add_argument("--seed", type=int, help="Seed for reproducibility.")
    p.add_argument("--threads", type=int, help="CPU threads if relevant.")
    p.add_argument("--max-new-tokens", type=int, default=64, help="Completion length used for measurement.")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature used for measurement.")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=60)
    p.add_argument("--repeat-penalty", type=float, default=1.05)

    p.add_argument("--start", type=int, default=32, help="Starting n_batch.")
    p.add_argument("--max-n", type=int, default=4096, help="Upper bound for n_batch.")
    p.add_argument("--growth", type=float, default=2.0, help="Geometric growth factor for n_batch.")
    p.add_argument("--warmup-passes", type=int, default=1, help="Warm-up runs at conservative n_batch.")
    p.add_argument("--drop-tolerance", type=float, default=0.08, help="Stop when eval TPS falls by this fraction from the best.")
    p.add_argument("--objective", choices=["eval_tps", "prompt_tps", "wall_tps", "total_ms"], default="eval_tps",
                   help="Optimisation target. Use 'wall_tps' or 'total_ms' when timings are missing.")

    p.add_argument("--llama-verbose", action="store_true", help="Verbose llama.cpp initialisation.")
    p.add_argument("--verbose", action="store_true", help="Verbose probe log.")
    p.add_argument("--out", type=Path, help="Write all probe results to JSON.")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    """
    Determine the optimal `n_batch` value for a given llama.cpp model.

    This function probes the optimal `n_batch` value by measuring throughput on a fixed prompt.
    It reloads the model for each candidate `n_batch`, measures tokens/sec on a short completion,
    and stops when throughput drops beyond a tolerance or an error occurs.

    Parameters:
    - `argv`: Command-line arguments.

    Returns:
    - An integer indicating program exit status.
    """

    args = parse_args(argv)

    prompt, stops = _build_prompt(args.prompt, args.prompt_file, args.messages, args.chat_format)

    sample_cfg = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
        seed=args.seed,
        threads=args.threads
    )

    best, results = probe_optimal_n_batch(
        model_path=args.model,
        prompt=prompt,
        stops=stops,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        start=args.start,
        max_n=args.max_n,
        growth=args.growth,
        warmup_passes=args.warmup_passes,
        drop_tolerance=args.drop_tolerance,
        sample_cfg=sample_cfg,
        verbose=args.verbose or True,          # default to verbose table
        verbose_llama=args.llama_verbose,
        objective=args.objective
    )

    print(f"\nRecommended n_batch: {best}")
    if args.out:
        payload = {
            "model": args.model,
            "n_ctx": args.n_ctx,
            "best_n_batch": best,
            "results": [asdict(r) for r in results]
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote results to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
