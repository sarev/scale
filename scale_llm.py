#!/usr/bin/env python3
"""
This module TODO.
"""

from __future__ import annotations

from dataclasses import dataclass
from llama_cpp import Llama
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Iterable
import math
import os
import re


Message = Dict[str, str]
Messages = List[Message]
Chunk = List[str]


def _strip_none(xs: Sequence[Optional[str]]) -> List[str]:
    """Remote all `None` values from a sequence of optional strings."""
    return [x for x in xs if x]


def _normalise_messages(messages: Messages) -> Messages:
    """
    Defensive normalisation. Keeps only known keys, coerces to strings.

    This function ensures that the messages dictionary contains only valid and expected key-value pairs.
    It strips any leading or trailing whitespace from the 'role' and 'content' fields and discards any
    messages where either field is empty.

    Parameters:
    - messages: The input list of message dictionaries to be normalised.

    Returns:
    - out: A new list of message dictionaries with only valid and normalised entries.

    Notes:
    - This function assumes that the input messages are dictionaries. If this assumption is violated,
      it will raise a TypeError.
    """

    out: Messages = []
    for m in messages:
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", "")).strip()
        if role == "" or content == "":
            continue
        out.append({"role": role, "content": content})
    return out


def format_chat_qwen(messages: Messages) -> Tuple[str, Chunk]:
    """
    Qwen2.5 ChatML format:

    <|im_start|>system
    ...<|im_end|>
    <|im_start|>user
    ...<|im_end|>
    <|im_start|>agent
    ...<|im_end|>
    <|im_start|>assistant
    ...<|im_end|>

    Returns (prompt, stop_tokens).
    """
    parts: Chunk = []
    for m in _normalise_messages(messages):
        role = m["role"]
        if role in ("system", "user", "agent", "assistant"):
            parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")  # open assistant turn as anchor
    prompt = "".join(parts)
    stops = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
    return prompt, stops


def format_chat_llama3(messages: Messages) -> Tuple[str, Chunk]:
    """
    Llama 3 Instruct style with header tokens.

    <|start_header_id|>system<|end_header_id|>

    ...<|eot_id|><|start_header_id|>user<|end_header_id|>

    ...<|eot_id|><|start_header_id|>agent<|end_header_id|>

    ...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    ...<|eot_id|>

    We open an assistant header without the trailing <|eot_id|>.
    """
    def block(role: str, content: str) -> str:
        return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    parts: Chunk = []
    for m in _normalise_messages(messages):
        role = m["role"]
        if role not in ("system", "user", "assistant", "agent"):
            continue
        parts.append(block(role, m["content"]))
    # open assistant header for generation
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    prompt = "".join(parts)
    stops = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]
    return prompt, stops


def format_chat_llama2(messages: Messages) -> Tuple[str, Chunk]:
    """
    Llama 2 Chat style using [INST] and optional <<SYS>>.

    If the first message is system, it is folded into the first user turn per Meta guide.
    """
    msgs = _normalise_messages(messages)
    sys_txt = ""
    if msgs and msgs[0]["role"] == "system":
        sys_txt = msgs[0]["content"]
        msgs = msgs[1:]

    parts: Chunk = []
    buf_user: Optional[str] = None

    for m in msgs:
        if m["role"] == "user" or m["role"] == "agent":
            if buf_user is not None:
                # Flush previous incomplete user block defensively
                parts.append(f"[INST] {buf_user.strip()} [/INST]")
            buf_user = m["content"]
        elif m["role"] == "assistant":
            # Close the pending user block, then add assistant completion
            user_payload = buf_user or ""
            if sys_txt:
                user_payload = f"<<SYS>>\n{sys_txt}\n<</SYS>>\n\n{user_payload}"
                sys_txt = ""  # only applied to the first user block
            parts.append(f"[INST] {user_payload.strip()} [/INST]{m['content']}")
            buf_user = None

    # Open an assistant turn
    user_payload = buf_user or ""
    if sys_txt:
        user_payload = f"<<SYS>>\n{sys_txt}\n<</SYS>>\n\n{user_payload}"
    parts.append(f"[INST] {user_payload.strip()} [/INST]")

    prompt = "".join(parts)
    stops = ["</s>", "[INST]"]
    return prompt, stops


def format_chat_mistral(messages: Messages) -> Tuple[str, Chunk]:
    """
    Mistral/Mixtral Instruct uses [INST] ... [/INST] blocks without <<SYS>> by default.
    """
    msgs = _normalise_messages(messages)
    parts: Chunk = []
    buf_user: Optional[str] = None

    for m in msgs:
        if m["role"] == "user" or m["role"] == "agent":
            if buf_user is not None:
                parts.append(f"[INST] {buf_user.strip()} [/INST]")
            buf_user = m["content"]
        elif m["role"] == "assistant":
            user_payload = buf_user or ""
            parts.append(f"[INST] {user_payload.strip()} [/INST]{m['content']}")
            buf_user = None

    parts.append(f"[INST] {(buf_user or '').strip()} [/INST]")
    prompt = "".join(parts)
    stops = ["</s>", "[INST]"]
    return prompt, stops


def format_chat_phi3(messages: Messages) -> Tuple[str, Chunk]:
    """
    Phi-3 style:

    <|system|>
    ...<|end|>
    <|user|>
    ...<|end|>
    <|agent|>
    ...<|end|>
    <|assistant|>
    ...<|end|>

    Open assistant tag without <|end|> to anchor generation.
    """
    parts: Chunk = []
    for m in _normalise_messages(messages):
        role = m["role"]
        if role == "system":
            parts.append(f"<|system|>\n{m['content']}<|end|>\n")
        elif role == "user":
            parts.append(f"<|user|>\n{m['content']}<|end|>\n")
        elif role == "agent":
            parts.append(f"<|agent|>\n{m['content']}<|end|>\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{m['content']}<|end|>\n")
    parts.append("<|assistant|>\n")
    prompt = "".join(parts)
    stops = ["<|end|>", "<|user|>", "<|system|>"]
    return prompt, stops


# Map normalised format keys to formatter callables.
FORMATTERS = {
    "qwen": format_chat_qwen,
    "chatml": format_chat_qwen,       # alias
    "llama3": format_chat_llama3,
    "llama-3": format_chat_llama3,    # alias
    "llama2": format_chat_llama2,
    "llama-2": format_chat_llama2,    # alias
    "mistral": format_chat_mistral,
    "mixtral": format_chat_mistral,   # alias
    "phi3": format_chat_phi3,
}


def llm_formatters() -> List[str]:
    return FORMATTERS.keys()


def _auto_detect_format(model_path: str) -> str:
    """
    Light heuristic based on filename. Returns a key present in FORMATTERS.
    Defaults to 'qwen' if unsure to avoid silent mismatches.
    """
    name = os.path.basename(model_path).lower()
    if re.search(r"qwen", name):
        return "qwen"
    if re.search(r"llama[-_ ]?3", name):
        return "llama3"
    if re.search(r"llama", name):
        return "llama2"
    if re.search(r"(mistral|mixtral)", name):
        return "mistral"
    if re.search(r"phi[-_ ]?3", name):
        return "phi3"
    return "qwen"


# ---------------------------- Runner class ----------------------------


@dataclass
class GenerationConfig:
    """
    Settings that control how text is generated. Values are safe defaults for small 7B models.
    All counts refer to tokens, not characters.

    Attributes
    ----------
    max_new_tokens : int
        Hard limit on how many tokens to generate after the prompt. Generation may stop earlier
        if an end-of-sequence token or a stop string is produced. Larger values take longer and
        use more memory for the KV cache.

    temperature : float
        Controls randomness by smoothing the probability distribution. 0 gives greedy decoding
        and is deterministic. Higher values increase diversity. Typical ranges:
        0.0–0.3 for factual or tool use, 0.7–1.0 for creative writing.

    top_p : float
        Nucleus sampling. At each step keep the smallest set of tokens whose cumulative
        probability ≥ top_p, then sample from that set. 1.0 effectively disables nucleus
        sampling. Lower values (for example 0.8–0.95) reduce riskier tokens.

    top_k : int
        Top-k sampling. At each step consider only the top_k most likely tokens. 0 usually
        disables this cap in most runtimes. Values in the 20–100 range are common. If both
        top_p and top_k are set, most backends apply both filters, which is more restrictive.

    repeat_penalty : float
        Penalises tokens that have appeared recently to reduce repetition. 1.0 means no penalty.
        Values slightly above 1.0, such as 1.05–1.2, are typical. Excessive penalties can harm
        fluency or cause early stops.

    Notes
    -----
    - Tokens are model-specific. 512 tokens is roughly 350–450 English words.
    - For maximum determinism: temperature=0, top_p=1.0, repeat_penalty≈1.0.
    - If outputs become terse or repetitive, relax one control at a time
      (increase temperature, raise top_p, or lower repeat_penalty).
    """

    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 60
    repeat_penalty: float = 1.05


class LocalChatModel:
    """
    Wraps llama_cpp.Llama with explicit chat formatting and raw completion.

    You can set `chat_format` to any key in FORMATTERS or "auto".
    """

    @classmethod
    def download_model(cls, repo_id: str, include: Union[str, Iterable[str]], models_path: str = "./models") -> Path:
        """
        Download a LLM model (.gguf file or files) into a local models folder.

        - Downloads into `<models_path>/<basename(repo_id)>`.
        - Filters with `include` (same idea as `--include`).
        - Returns the path to the downloaded file, or the first shard if present (`*-00001-of-*.gguf`).
        - `models_path` is relative to CWD or absolute; the returned path will be likewise.

        Parameters
        ----------
        repo_id : str
            Hub repository ID, e.g. "Qwen/Qwen2.5-7B-Instruct-GGUF".
        include : str | Iterable[str]
            Glob pattern(s) to include (mirrors `--include`), e.g. "qwen2.5-7b-instruct-*.gguf".
        models_path : str, optional
            Base directory for all models. Defaults to "./models". A subdirectory named after the
            basename of `repo_id` is created inside this directory.

        Returns
        -------
        pathlib.Path
            Path to the selected file (relative or absolute per policy above).

        Raises
        ------
        FileNotFoundError
            If nothing matches.

        Examples
        --------

        Find the LLM repository you want at https://huggingface.co/ (and generally request access) then locate
        the "Files and versions" to see how the model file(s) is/are named. E.g. "*.safetensors" or "*.gguf".

        model_file = LocalChatModel.download_model("Qwen/Qwen2.5-7B-Instruct-GGUF", " qwen2.5-7b-instruct-q4_k_m-*.gguf", "/d/Programming/llm/models")

        model_file = LocalChatModel.download_model("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf")

        """

        from huggingface_hub import snapshot_download

        patterns = [include] if isinstance(include, str) else list(include)
        base = Path(models_path).expanduser()
        out = base / repo_id
        out.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(out),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=patterns
        )

        files = sorted({p for pat in patterns for p in out.rglob(pat) if p.is_file()})
        if not files:
            raise FileNotFoundError(f"No files matched {patterns} in {out}")

        first = next((p for p in files if re.search(r"-0*1-of-0*\d+\.gguf$", p.name, re.I)), None)
        chosen = first or next((p for p in files if p.suffix.lower() == ".gguf"), files[0])

        return chosen if not base.is_absolute() else chosen.resolve()

    def __init__(
        self,
        model_path: str,
        *,
        chat_format: Optional[str] = None,
        n_ctx: int = 32 * 1024,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        **llama_kwargs,
    ) -> None:
        """
        Initialise a local GGUF model via llama.cpp and select a chat formatting scheme for prompt construction.

        Parameters
        ----------
        model_path : str
            Filesystem path to the GGUF model. The file must exist. This value is passed as `model_path` to
            `llama_cpp.Llama`.
        chat_format : Optional[str], default None
            Chat formatting key or "auto". When "auto" or None, a light heuristic uses the model filename to
            pick one of: {"qwen", "llama3", "llama2", "mistral", "phi3"}.
            You may override later with `set_chat_format`.
        n_ctx : int, default 8192
            Context window for the model in tokens. Forwarded to `llama_cpp.Llama(n_ctx=...)`.
        n_batch : int, default 512
            Prompt processing batch size in tokens. Forwarded to `llama_cpp.Llama(n_batch=...)`.
        n_gpu_layers : int, default -1
            Number of layers to offload to GPU. Use -1 to offload as many as fit, or 0 for CPU only. Forwarded
            to `llama_cpp.Llama`.
        verbose : bool, default False
            Verbosity flag passed to `llama_cpp.Llama(verbose=...)`.
        **llama_kwargs
            Any additional keyword arguments forwarded directly to `llama_cpp.Llama`, for example `seed`,
            `n_threads`, `rope_scaling_type`, or backend specific flags.

        Attributes set
        --------------
        chat_format : str
            The selected chat formatting key after resolution.
        model_path : str
            The path passed in.
        llm : llama_cpp.Llama
            The loaded model instance.

        Raises
        ------
        FileNotFoundError
            If `model_path` does not exist.
        ValueError
            If an explicit `chat_format` is provided but not recognised.
        Exception
            Any error propagated from `llama_cpp.Llama` during model loading.

        Notes
        -----
        Auto detection is filename based. It looks for substrings such as "qwen", "llama3", "llama", "mistral"
        or "mixtral", and "phi3". If no match is found it defaults to "qwen".
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Resolve format eagerly to fail fast if invalid
        auto_fmt = _auto_detect_format(model_path)
        chosen = (chat_format or "auto").lower()
        if chosen == "auto":
            chosen = auto_fmt
        if chosen not in FORMATTERS:
            raise ValueError(
                f"Unknown chat_format '{chat_format}'. "
                f"Valid: {sorted(FORMATTERS.keys())}"
            )

        self.chat_format: str = chosen
        self.model_path = model_path

        # Create the llama.cpp model
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=verbose,
            **llama_kwargs,
        )

        self.n_ctx = n_ctx                      # Size of the context window in tokens
        self.approx_bpt = 2.0                   # Estimate of the number of raw bytes per token
        self.turn = 0                           # Current turn number (incremented after each assistent turn)
        self.refine_every = 8                   # Refine our bytes per token every this many turns
        self.ctx_margin = int(n_ctx * 0.10)     # Safety margin in calculation for whether the prompt risks exceeding the context window

    def set_chat_format(self, chat_format: str) -> None:
        """Allow manual override of the chat format for this instance."""
        key = chat_format.lower()
        if key not in FORMATTERS:
            raise ValueError(f"Unknown chat_format '{chat_format}'. Valid: {sorted(FORMATTERS.keys())}")
        self.chat_format = key

    def _build_prompt(self, messages: Messages) -> Tuple[str, Chunk]:
        formatter = FORMATTERS[self.chat_format]
        return formatter(messages)

    def _trim_needed(self, cfg: GenerationConfig, prompt: str):
        # Get prompt text as raw UTF-8 bytes
        prompt_utf8 = prompt.encode("utf-8")
        prompt_bytes = len(prompt_utf8)

        # Estimate the number of tokens in the total prompt
        approx_tokens = math.ceil(prompt_bytes / self.approx_bpt)

        # Are we refining our estimate of the number of bytes per token?
        if self.turn % self.refine_every == 0:
            true_tokens = len(self.llm.tokenize(prompt_utf8, add_bos=True))
            self.approx_bpt = prompt_bytes / true_tokens

        return approx_tokens > (self.n_ctx - self.ctx_margin)

    def generate(
        self,
        messages: Messages,
        *,
        cfg: Optional[GenerationConfig] = None,
        stop: Optional[Chunk] = None,
    ) -> str:
        """
        Produce a single non-streaming completion using an explicit chat-formatted prompt. The method
        builds a prompt for the selected `chat_format`, opens an assistant turn to anchor generation, and
        calls `llama_cpp.Llama.create_completion`.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            Ordered chat transcript as dictionaries with keys {"role", "content"}. Expected roles are
            "system", "user", and "assistant". Unknown roles are ignored by formatters. The list must be
            non-empty.
        cfg : Optional[GenerationConfig], default None
            Generation settings. If None, defaults are used:
            `max_new_tokens=512`, `temperature=0.3`, `top_p=0.9`, `top_k=60`, `repeat_penalty=1.05`.
            These map to the corresponding `create_completion` parameters.
        stop : Optional[Chunk], default None
            Additional stop strings to append after the formatter specific stop set. Duplicates are removed
            with first occurrence preserved. Use this to prevent the model from starting new turns that are
            specific to your application protocol.

        Returns
        -------
        str
            The assistant text returned by llama.cpp for the open assistant turn, with trailing whitespace
            stripped.

        Raises
        ------
        ValueError
            If `messages` is not a non-empty list of role and content pairs.
        Exception
            Any error propagated from `llama_cpp.Llama.create_completion`.

        Behaviour
        ---------
        Each formatter supplies a conservative default stop list to avoid the model emitting role markers or
        end-of-text tokens. You can pass a `seed` via `**llama_kwargs` at construction time if you need
        reproducibility. The call is synchronous and does not stream tokens.
        """

        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list of {'role','content'} dicts.")

        prompt, default_stops = self._build_prompt(messages)

        user_stops = stop or []
        # De-duplicate while preserving order
        seen: set = set()
        all_stops: Chunk = []
        for tok in list(default_stops) + list(user_stops):
            if tok and tok not in seen:
                all_stops.append(tok)
                seen.add(tok)

        cfg = cfg or GenerationConfig()

        if self._trim_needed(cfg, prompt):
            print("WARNING: we're getting close to the context window limit!")

        resp = self.llm.create_completion(
            prompt=prompt,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=all_stops,
        )

        # Increment our count of the number of LLM turns there have been
        self.turn += 1

        # llama.cpp returns plain text for completion choices
        text = resp.get("choices", [{}])[0].get("text", "")
        return text.rstrip()

    def progressive_generate(
        self,
        messages: List[Dict[str, str]],
        *,
        cfg: Optional[GenerationConfig] = None,
        stop: Optional[Chunk] = None,
    ):
        """
        Stream a completion progressively as an iterator of text fragments.

        This is a generator that yields incremental text pieces from
        `llama_cpp.Llama.create_completion(stream=True)` so that callers can render output as it
        is produced.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            Ordered chat transcript with items shaped like
            {"role": "system"|"user"|"assistant", "content": str}.
            Unknown roles are ignored by the selected formatter. Must be non-empty.
        cfg : Optional[GenerationConfig], default None
            Generation settings. If None, defaults are used:
            `max_new_tokens=512`, `temperature=0.3`, `top_p=0.9`,
            `top_k=60`, `repeat_penalty=1.05`.
        stop : Optional[Chunk], default None
            Additional stop strings to append after the formatter defaults.
            Duplicates are removed with first occurrence preserved.

        Yields
        ------
        str
            A non-empty text fragment for immediate display. Concatenate all yielded fragments to obtain
            the full completion. Trailing whitespace is not trimmed.

        Raises
        ------
        ValueError
            If `messages` is not a non-empty list of role and content pairs.
        Exception
            Any error propagated from `llama_cpp.Llama.create_completion`.

        Notes
        -----
        - The final chunk may be empty if the model stops on a stop token.
        - Whitespace and newlines are yielded as produced by the model.
        - If you need a final trimmed string, collect fragments then `rstrip()`.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list of {'role','content'} dicts.")

        prompt, default_stops = self._build_prompt(messages)

        user_stops = stop or []
        seen: set = set()
        all_stops: Chunk = []
        for tok in list(default_stops) + list(user_stops):
            if tok and tok not in seen:
                all_stops.append(tok)
                seen.add(tok)

        cfg = cfg or GenerationConfig()

        if self._trim_needed(cfg, prompt):
            print("WARNING: we're getting close to the context window limit!")

        stream = self.llm.create_completion(
            prompt=prompt,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=all_stops,
            stream=True,
        )

        # Increment our count of the number of LLM turns there have been
        self.turn += 1

        for chunk in stream:
            # llama.cpp streams small deltas; skip empty pieces defensively.
            piece = chunk.get("choices", [{}])[0].get("text", "")
            if piece:
                yield piece

    def get_supported_formats(self) -> list[str]:
        """
        Return the list of supported chat format keys.

        Returns
        -------
        list[str]
            Sorted list of recognised chat formatter names that this instance can use. These are the keys
            accepted by `set_chat_format` and the `chat_format` constructor argument.
        """
        return sorted(FORMATTERS.keys())

    def detect_chat_format(self, model_path: str | None = None) -> str:
        """
        Detect the most likely chat format for a model filename.

        Parameters
        ----------
        model_path : str | None, default None
            Filesystem path or filename used for heuristic detection. If omitted, the instance's
            `self.model_path` is used.

        Returns
        -------
        str
            A chat format key present in `get_supported_formats()`.

        Notes
        -----
        This uses the same filename-based heuristic as the constructor: it looks for substrings like "qwen",
        "llama3", "llama", "mistral" or "mixtral", and "phi3". If no match is found, it defaults to "qwen".
        """
        target = model_path or self.model_path
        return _auto_detect_format(target)

    def count_tokens(self, messages: Messages) -> int:
        """
        Count the number of tokens the model would consume for the formatted prompt.

        The messages are first converted to a prompt using the current `chat_format`, including the open
        assistant anchor. Only the prompt tokens are counted. Generated tokens are not included.

        Parameters
        ----------
        messages : Messages
            Ordered chat transcript as a list of dicts with keys {"role", "content"}.
            Expected roles are "system", "user", and "assistant". The list must be non-empty.

        Returns
        -------
        int
            The number of tokens for the formatted prompt, as tokenised by the
            underlying model's tokenizer.

        Raises
        ------
        ValueError
            If `messages` is not a non-empty list.

        Notes
        -----
        Special tokens such as ChatML markers are counted as single tokens when the tokenizer supports it. If the
        installed `llama_cpp` version does not accept the `special=True` argument, the method falls back to
        standard tokenisation.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list of {'role','content'} dicts.")

        prompt, _ = self._build_prompt(messages)

        # Prefer counting with special tokens enabled. Fall back if unsupported.
        data = prompt.encode("utf-8")
        try:
            ids = self.llm.tokenize(data, add_bos=True, special=True)  # type: ignore[arg-type]
        except TypeError:
            ids = self.llm.tokenize(data, add_bos=True)  # type: ignore[arg-type]
        return len(ids)
