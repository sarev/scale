#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

SCALE's model layer: `LocalChatModel` wraps a local GGUF model loaded through `llama_cpp.Llama`, offering blocking
(`generate`) and streaming (`progressive_generate`) chat completion behind one interface, with stop-token merging and
a context-budget check on every call. `GenerationConfig` carries the sampling and length settings for a single
generation.

A family of formatters renders message lists into the prompt shapes the supported model families expect - Qwen ChatML,
Llama 3, Llama 2, Mistral and Phi-3 - and the right one is auto-detected from the model filename unless named
explicitly.

Token budgeting uses a cheap bytes-per-token estimate that is periodically recalibrated against the real tokeniser, so
prompt sizing stays accurate without tokenising every turn; `snippet_budget` works out how much source can still fit
alongside a conversation, and `download_model` fetches a GGUF snapshot from Hugging Face.
"""

from __future__ import annotations

from dataclasses import dataclass
from llama_cpp import Llama
from pathlib import Path
from scale_log import echo
from typing import Dict, List, Optional, Sequence, Tuple, Union, Iterable
import math
import os
import re


Message = Dict[str, str]
Messages = List[Message]
Chunk = List[str]


# When budgeting how much of a routine snippet may be sent to the model, this many tokens are held back
# as headroom for the comment the model will generate in reply. A comment is short, so reserving the full
# `max_new_tokens` (often several thousand) would needlessly shrink the snippet budget. See
# `LocalChatModel.snippet_budget`.
COMMENT_GENERATION_RESERVE = 1024


def _strip_none(xs: Sequence[Optional[str]]) -> List[str]:
    """
    Filter a sequence of optional strings down to its truthy entries.

    Parameters:
    - `xs`: The sequence to filter; `None` and empty strings are both dropped.

    Returns:
    - A list of the remaining non-empty strings.
    """

    return [x for x in xs if x]


def _normalise_messages(messages: Messages) -> Messages:
    """
    Return a cleaned copy of a chat message list.

    Roles and contents are coerced to stripped strings, and any message with a blank role or blank content is dropped so the chat formatters never emit empty turns.

    Parameters:
    - `messages`: The list of role/content message dictionaries.

    Returns:
    - A new list of normalised messages; the input is left unmodified.
    """

    out: Messages = []

    # Drop any message with a blank role or content so the chat formatters never emit empty turns.
    for m in messages:
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", "")).strip()
        if role == "" or content == "":
            continue
        out.append({"role": role, "content": content})

    return out


def format_chat_qwen(messages: Messages) -> Tuple[str, Chunk]:
    """
    Render chat messages into a Qwen ChatML prompt.

    Each recognised role is wrapped in `<|im_start|>`/`<|im_end|>` markers, and an unclosed assistant header is appended so the model continues as the assistant.

    Parameters:
    - `messages`: The list of role/content message dictionaries.

    Returns:
    - A tuple of the prompt string and the list of stop strings that end generation.
    """

    parts: Chunk = []

    # Each recognised role becomes one ChatML turn; unknown roles are dropped silently.
    for m in _normalise_messages(messages):
        role = m["role"]
        if role in ("system", "user", "agent", "assistant"):
            parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>")

    # The unclosed assistant header anchors generation; the stop strings end it at the next ChatML marker.
    parts.append("<|im_start|>assistant\n")  # open assistant turn as anchor
    prompt = "".join(parts)
    stops = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]

    return prompt, stops


def format_chat_llama3(messages: Messages) -> Tuple[str, Chunk]:
    """
    Render chat messages into a Llama 3 instruct prompt.

    Each recognised role becomes a header/`<|eot_id|>` block, with an unclosed assistant header appended so the model replies as the assistant.

    Parameters:
    - `messages`: The list of role/content message dictionaries.

    Returns:
    - A tuple of the prompt string and the list of stop strings that end generation.
    """

    def block(role: str, content: str) -> str:
        """
        Wrap one message in Llama 3 header and end-of-turn tokens.

        Parameters:
        - `role`: The speaker role placed in the header.
        - `content`: The message text.

        Returns:
        - The message rendered as a single `<|start_header_id|>`...`<|eot_id|>` segment.
        """

        return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    parts: Chunk = []

    # Only recognised roles are rendered; anything else is skipped rather than guessed at.
    for m in _normalise_messages(messages):
        role = m["role"]
        if role not in ("system", "user", "assistant", "agent"):
            continue
        parts.append(block(role, m["content"]))

    # The dangling assistant header prompts the model to reply; the stop list ends it at the next turn marker.
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    prompt = "".join(parts)
    stops = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]

    return prompt, stops


def format_chat_llama2(messages: Messages) -> Tuple[str, Chunk]:
    """
    Render a chat transcript as a single Llama 2 `[INST]`-style prompt string.

    Llama 2 has no system turn, so a leading system message is folded into the first user block via the `<<SYS>>` convention. User/agent turns are paired with their assistant replies, and the prompt always ends with an open `[INST]` block to cue the model's next reply.

    Parameters:
    - `messages`: The conversation as a list of `{'role', 'content'}` dicts.

    Returns:
    - A tuple of the formatted prompt string and the default stop strings for this format.
    """

    msgs = _normalise_messages(messages)
    sys_txt = ""

    # Llama 2 has no system turn: peel off a leading system message for inlining into the first user block.
    if msgs and msgs[0]["role"] == "system":
        sys_txt = msgs[0]["content"]
        msgs = msgs[1:]

    # buf_user holds the latest user turn until an assistant reply pairs with it.
    parts: Chunk = []
    buf_user: Optional[str] = None

    # Pair each user/agent turn with its assistant reply; a second user turn arriving first flushes the pending one as its own [INST] block.
    for m in msgs:
        if m["role"] == "user" or m["role"] == "agent":
            if buf_user is not None:
                parts.append(f"[INST] {buf_user.strip()} [/INST]")
            buf_user = m["content"]
        elif m["role"] == "assistant":
            user_payload = buf_user or ""

            # Per the <<SYS>> convention the system text is injected into the first user payload only.
            if sys_txt:
                user_payload = f"<<SYS>>\n{sys_txt}\n<</SYS>>\n\n{user_payload}"
                sys_txt = ""  # only applied to the first user block

            # The assistant reply follows [/INST] verbatim, replaying past turns as context.
            parts.append(f"[INST] {user_payload.strip()} [/INST]{m['content']}")
            buf_user = None

    # Always close with an open [INST] block (catching any unconsumed system text) so the prompt invites the next reply.
    user_payload = buf_user or ""
    if sys_txt:
        user_payload = f"<<SYS>>\n{sys_txt}\n<</SYS>>\n\n{user_payload}"
    parts.append(f"[INST] {user_payload.strip()} [/INST]")
    prompt = "".join(parts)
    stops = ["</s>", "[INST]"]

    return prompt, stops


def format_chat_mistral(messages: Messages) -> Tuple[str, Chunk]:
    """
    Render a chat transcript into the Mistral/Mixtral `[INST]` prompt format.

    User and agent turns are buffered and paired with the assistant reply that consumes them; a final `[INST]` block (empty if nothing is pending) always closes the prompt so the model continues as the assistant. System messages have no slot in this template and are not emitted.

    Parameters:
    - `messages`: The conversation as a list of `{'role', 'content'}` dicts.

    Returns:
    - A tuple of the formatted prompt string and the stop tokens for this format.
    """

    # Hold the latest user/agent turn in a buffer so it can be paired with the assistant reply that follows.
    msgs = _normalise_messages(messages)
    parts: Chunk = []
    buf_user: Optional[str] = None

    # Pair each buffered turn with its assistant reply; back-to-back user turns flush as separate [INST] blocks.
    for m in msgs:
        if m["role"] == "user" or m["role"] == "agent":
            if buf_user is not None:
                parts.append(f"[INST] {buf_user.strip()} [/INST]")
            buf_user = m["content"]
        elif m["role"] == "assistant":
            user_payload = buf_user or ""
            parts.append(f"[INST] {user_payload.strip()} [/INST]{m['content']}")
            buf_user = None

    # Always close with a final [INST] block - empty if nothing is pending - so the model continues as the assistant.
    parts.append(f"[INST] {(buf_user or '').strip()} [/INST]")
    prompt = "".join(parts)
    stops = ["</s>", "[INST]"]

    return prompt, stops


def format_chat_phi3(messages: Messages) -> Tuple[str, Chunk]:
    """
    Render a chat transcript as a Phi-3 tagged prompt string.

    Each message is wrapped in its matching `<|role|>`...`<|end|>` tag (messages with unrecognised roles are silently dropped), and a trailing `<|assistant|>` tag cues the model to reply.

    Parameters:
    - `messages`: The conversation as a list of `{'role', 'content'}` dicts.

    Returns:
    - A tuple of the formatted prompt string and the default stop strings for this format.
    """

    parts: Chunk = []

    for m in _normalise_messages(messages):
        role = m["role"]

        # Only the four known roles are emitted; anything else is silently dropped.
        if role == "system":
            parts.append(f"<|system|>\n{m['content']}<|end|>\n")
        elif role == "user":
            parts.append(f"<|user|>\n{m['content']}<|end|>\n")
        elif role == "agent":
            parts.append(f"<|agent|>\n{m['content']}<|end|>\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{m['content']}<|end|>\n")

    # Trailing <|assistant|> tag cues the model to start its reply.
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
    """
    List the names of the supported chat prompt formats.

    Returns:
    - The accepted `chat_format` keys (a live view over the formatter registry, despite the `List[str]` annotation).
    """

    # Returns the live dict keys view, not a list, despite the annotation.
    return FORMATTERS.keys()


def _auto_detect_format(model_path: str) -> str:
    """
    Guess the chat prompt format from a model file's name.

    Patterns are tried in order, with the specific `llama3` match tested before the generic `llama` fallback; unrecognised names default to `qwen`.

    Parameters:
    - `model_path`: Path to the GGUF model file.

    Returns:
    - A formatter key registered in `FORMATTERS` (defaults to `qwen`).
    """

    # Order matters: llama3 must match before the bare llama fallback; unknown names default to qwen.
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
    Sampling and length settings for one generation call.

    Defaults favour focused, low-temperature output with a mild repeat penalty, suited to comment generation; pass an instance to `generate` or `progressive_generate` to override per call.
    """

    # Conservative defaults: low temperature plus a mild repeat penalty keep output focused.
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 60
    repeat_penalty: float = 1.05


class LocalChatModel:

    """
    A local GGUF chat model wrapped around `llama_cpp.Llama`.

    Builds prompts for several chat formats (auto-detected from the model filename unless given explicitly), tracks the context budget with a cheap bytes-per-token estimate that is periodically recalibrated, and offers both blocking and streaming generation.
    """

    # Class-level helper: fetch a model from Hugging Face before any instance exists.
    @classmethod
    def download_model(cls, repo_id: str, include: Union[str, Iterable[str]], models_path: str = "./models") -> Path:
        """
        Download a model snapshot from Hugging Face and return the path of its main GGUF file.

        Only files matching the include patterns are fetched (resumable), and for multi-part GGUFs the first shard is preferred so the result can be passed straight to the loader.

        Parameters:
        - `repo_id`: The Hugging Face repository identifier.
        - `include`: A glob pattern, or iterable of patterns, selecting the files to download.
        - `models_path`: Local directory under which the snapshot is stored.

        Returns:
        - Path to the chosen GGUF file, kept relative when `models_path` is relative; raises `FileNotFoundError` if nothing matched.
        """

        # Fetch only the matching files, then prefer the first shard of a multi-part GGUF as the loadable entry point.
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

        # A relative models_path stays relative so cwd-anchored default paths survive; absolute ones are fully resolved.
        return chosen if not base.is_absolute() else chosen.resolve()

    # Resolve the chat format up front ('auto' falls back to the filename heuristic), then load the weights.
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
        Load a GGUF model with `llama.cpp` and bind it to a chat prompt formatter.

        The chat format defaults to `auto`, in which case it is inferred from the model filename, and an unrecognised name is rejected before the costly model load. The remaining attributes seed the adaptive bytes-per-token estimate used to budget prompts against the context window.

        Parameters:
        - `model_path`: Path to the GGUF model file.
        - `chat_format`: Prompt formatter name, or `None`/`"auto"` to infer it from the filename.
        - `n_ctx`: Context window size in tokens.
        - `n_batch`: Prompt-evaluation batch size.
        - `n_gpu_layers`: Number of layers to offload to the GPU (`-1` for all).
        - `verbose`: Verbosity flag passed through to `llama_cpp`.
        - `llama_kwargs`: Extra keyword arguments forwarded to the `Llama` constructor.

        Notes:
        This raises `FileNotFoundError` if `model_path` does not exist, or `ValueError` for an unknown chat format.
        """

        # Resolve `auto` to a chat format inferred from the model filename.
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        auto_fmt = _auto_detect_format(model_path)
        chosen = (chat_format or "auto").lower()
        if chosen == "auto":
            chosen = auto_fmt

        # Reject unknown formats up front, before the costly model load.
        if chosen not in FORMATTERS:
            raise ValueError(
                f"Unknown chat_format '{chat_format}'. "
                f"Valid: {sorted(FORMATTERS.keys())}"
            )

        # Load the model, then seed the adaptive bytes-per-token estimate used for context budgeting.
        self.chat_format: str = chosen
        self.model_path = model_path
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

    # Switch prompt formats on a live model without reloading the weights.
    def set_chat_format(self, chat_format: str) -> None:
        """
        Select the chat template used to render message lists into prompts.

        The name is matched case-insensitively against the registered formatters; an unknown name raises `ValueError` listing the valid choices.

        Parameters:
        - `chat_format`: Name of a registered chat formatter (case-insensitive).
        """

        # Lower-case before the lookup so format names are case-insensitive.
        key = chat_format.lower()
        if key not in FORMATTERS:
            raise ValueError(f"Unknown chat_format '{chat_format}'. Valid: {sorted(FORMATTERS.keys())}")
        self.chat_format = key

    def _build_prompt(self, messages: Messages) -> Tuple[str, Chunk]:
        """
        Render a message list into a single prompt string via the active chat formatter.

        Parameters:
        - `messages`: The conversation as a list of `{'role', 'content'}` dicts.

        Returns:
        - A tuple of the formatted prompt string and the formatter's default stop tokens.
        """

        formatter = FORMATTERS[self.chat_format]
        return formatter(messages)

    # Cheap bytes-per-token estimate, recalibrated against a real tokenize call every refine_every turns.
    def _estimate_prompt_tokens(self, prompt: str) -> int:
        """
        Estimate the token count of a prompt, periodically recalibrating against the real tokeniser.

        Most turns use a cheap bytes-per-token approximation; every `refine_every` turns the prompt is tokenised for real and the ratio refreshed, keeping estimates accurate without paying for full tokenisation on every call.

        Parameters:
        - `prompt`: The fully rendered prompt text.

        Returns:
        - The estimated token count, or the exact count on calibration turns.
        """

        # Work in UTF-8 bytes: the bytes-per-token ratio is calibrated against them.
        prompt_utf8 = prompt.encode("utf-8")
        prompt_bytes = len(prompt_utf8)

        # Every `refine_every` turns, pay for a real tokenisation to recalibrate the cheap estimate.
        if self.turn % self.refine_every == 0:
            true_tokens = len(self.llm.tokenize(prompt_utf8, add_bos=True))

            # A zero token count would make the ratio update divide by zero.
            if true_tokens:
                # Refresh the calibration ratio while an exact count is in hand.
                self.approx_bpt = prompt_bytes / true_tokens
                return true_tokens

        # Between calibrations, the byte-ratio approximation is good enough.
        return math.ceil(prompt_bytes / self.approx_bpt)

    # Hard error only when the prompt alone cannot fit; a squeezed reply budget merely warns of possible truncation.
    def _check_context_budget(self, cfg: GenerationConfig, prompt: str) -> None:
        """
        Check a prompt against the context window, raising on overflow and warning on a tight fit.

        A prompt that exceeds the window outright raises `ValueError`, since generation could not succeed; one that merely leaves too little room for the configured reply length only triggers a warning, as the output may simply be truncated.

        Parameters:
        - `cfg`: Generation settings supplying `max_new_tokens`.
        - `prompt`: The fully rendered prompt text.
        """

        prompt_tokens = self._estimate_prompt_tokens(prompt)
        available = self.n_ctx - self.ctx_margin

        # A prompt that cannot fit at all is unrecoverable, so fail hard.
        if prompt_tokens >= self.n_ctx:
            raise ValueError(
                f"Prompt is ~{prompt_tokens} tokens but the context window (n_ctx) is only {self.n_ctx}. "
                f"Increase --n-ctx or process a smaller source file."
            )

        # A squeezed reply is merely truncated, so this case only warns.
        if prompt_tokens + cfg.max_new_tokens > available:
            echo(
                f"WARNING: prompt (~{prompt_tokens} tokens) plus max_new_tokens ({cfg.max_new_tokens}) "
                f"exceeds the usable context ({available} of {self.n_ctx}); generated output may be truncated."
            )

    # Estimate from the current bytes-per-token ratio; cheap but approximate, no tokenizer call.
    def estimate_tokens(self, text: str) -> int:
        """
        Cheaply estimate the token count of arbitrary text.

        Uses the bytes-per-token ratio maintained by the prompt estimator, so accuracy tracks whatever was last tokenised for real.

        Parameters:
        - `text`: The text to measure; may be empty.

        Returns:
        - The estimated token count, or 0 for empty text.
        """

        # Reuses the bytes-per-token ratio calibrated by the prompt estimator.
        if not text:
            return 0
        return math.ceil(len(text.encode("utf-8")) / self.approx_bpt)

    # Tokens of snippet that still fit once the margin, reply reserve, wrapper and existing messages are subtracted.
    def snippet_budget(
        self,
        messages: Messages,
        cfg: GenerationConfig,
        *,
        wrapper_reserve: int = 128,
        reserve: Optional[int] = None,
    ) -> int:
        """
        Work out how many tokens of source snippet can fit into a single turn.

        Starting from the full context window, this deducts the safety margin, room for the model's reply, a reserve for the prompt wrapper text, and the tokens already used by the conversation so far.

        Parameters:
        - `messages`: The conversation so far; its token count is deducted as overhead.
        - `cfg`: Generation settings whose `max_new_tokens` caps the reply reserve.
        - `wrapper_reserve`: Tokens set aside for the prompt scaffolding around the snippet.
        - `reserve`: Optional reply reserve; defaults to the comment-generation reserve.

        Returns:
        - The token budget available for the snippet; may be negative if the conversation is already oversized.
        """

        # Never reserve more reply room than the model is actually allowed to generate.
        reply_reserve = min(reserve if reserve is not None else COMMENT_GENERATION_RESERVE, cfg.max_new_tokens)
        overhead = self.count_tokens(messages) if messages else 0

        return self.n_ctx - self.ctx_margin - reply_reserve - wrapper_reserve - overhead

    # Blocking completion: merge default and caller stop tokens (defaults first, deduplicated) before a single create_completion call.
    def generate(
        self,
        messages: Messages,
        *,
        cfg: Optional[GenerationConfig] = None,
        stop: Optional[Chunk] = None,
    ) -> str:
        """
        Run one non-streaming chat completion and return the model's reply text.

        The message list is rendered through the active chat formatter, stop tokens from the formatter and the caller are merged, and the prompt is budget-checked against the context window before sampling.

        Parameters:
        - `messages`: Non-empty list of `{'role', 'content'}` dicts forming the conversation.
        - `cfg`: Optional generation settings; a default `GenerationConfig` is used if omitted.
        - `stop`: Optional extra stop tokens, merged with the formatter's defaults.

        Returns:
        - The generated reply with trailing whitespace stripped; may be empty.

        Notes:
        This raises `ValueError` if `messages` is invalid or the prompt cannot fit the context window.
        """

        # Render the conversation into a single prompt via the active chat formatter.
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list of {'role','content'} dicts.")
        prompt, default_stops = self._build_prompt(messages)
        user_stops = stop or []
        seen: set = set()
        all_stops: Chunk = []

        # Merge formatter and caller stop tokens, dropping blanks and duplicates while keeping order.
        for tok in list(default_stops) + list(user_stops):
            if tok and tok not in seen:
                all_stops.append(tok)
                seen.add(tok)

        # Budget-check, then run one blocking completion; the turn counter drives periodic token-estimate recalibration.
        cfg = cfg or GenerationConfig()
        self._check_context_budget(cfg, prompt)
        resp = self.llm.create_completion(
            prompt=prompt,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=all_stops,
        )
        self.turn += 1
        text = resp.get("choices", [{}])[0].get("text", "")

        # Models often pad replies with trailing newlines, so strip them.
        return text.rstrip()

    # Streaming twin of generate(): same prompt and stop handling, but yields text pieces as they arrive.
    def progressive_generate(
        self,
        messages: List[Dict[str, str]],
        *,
        cfg: Optional[GenerationConfig] = None,
        stop: Optional[Chunk] = None,
    ):
        """
        Stream a chat completion, yielding the reply text piece by piece as it is generated.

        This mirrors `generate` (same prompt rendering, stop-token merging and context budget check) but returns a generator of text fragments so callers can show progress live. Unlike `generate`, no trailing-whitespace trimming is applied.

        Parameters:
        - `messages`: Non-empty list of `{'role', 'content'}` dicts forming the conversation.
        - `cfg`: Optional generation settings; a default `GenerationConfig` is used if omitted.
        - `stop`: Optional extra stop tokens, merged with the formatter's defaults.

        Returns:
        - A generator yielding non-empty text fragments of the reply in order.

        Notes:
        This raises `ValueError` if `messages` is invalid or the prompt cannot fit the context window.
        """

        # Same validation and prompt rendering as `generate`.
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list of {'role','content'} dicts.")
        prompt, default_stops = self._build_prompt(messages)
        user_stops = stop or []
        seen: set = set()
        all_stops: Chunk = []

        # Merge formatter and caller stop tokens, as in `generate`.
        for tok in list(default_stops) + list(user_stops):
            if tok and tok not in seen:
                all_stops.append(tok)
                seen.add(tok)

        # Identical setup to `generate`, but the completion is opened in streaming mode.
        cfg = cfg or GenerationConfig()
        self._check_context_budget(cfg, prompt)
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
        self.turn += 1

        # Relay fragments as they arrive, skipping empty keep-alive chunks.
        for chunk in stream:
            piece = chunk.get("choices", [{}])[0].get("text", "")
            if piece:
                yield piece

    def get_supported_formats(self) -> list[str]:
        """
        Report the chat formats this wrapper knows how to apply.

        Returns:
        - A sorted list of the formatter names registered in `FORMATTERS`.
        """

        return sorted(FORMATTERS.keys())

    def detect_chat_format(self, model_path: str | None = None) -> str:
        """
        Detect the chat template format for a GGUF model.

        Parameters:
        - `model_path`: The model file to inspect; defaults to this instance's own model path.

        Returns:
        - The name of the detected chat format.
        """

        target = model_path or self.model_path
        return _auto_detect_format(target)

    # Exact tokenizer count; the fallback tolerates llama_cpp builds whose tokenize() lacks the 'special' argument.
    def count_tokens(self, messages: Messages) -> int:
        """
        Count the tokens a message list will occupy once formatted into a prompt.

        The messages are rendered through the same chat template used for generation, so the count includes the template's own formatting overhead.

        Parameters:
        - `messages`: A non-empty list of `{'role', 'content'}` dicts.

        Returns:
        - The number of tokens in the fully formatted prompt.

        Notes:
        Raises `ValueError` if `messages` is not a non-empty list.
        """

        # Render the messages through the real chat template so the count matches what generation will actually consume.
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list of {'role','content'} dicts.")
        prompt, _ = self._build_prompt(messages)
        data = prompt.encode("utf-8")

        # Older llama_cpp builds lack the `special` keyword, so fall back to the narrower signature.
        try:
            ids = self.llm.tokenize(data, add_bos=True, special=True)  # type: ignore[arg-type]
        except TypeError:
            ids = self.llm.tokenize(data, add_bos=True)  # type: ignore[arg-type]

        return len(ids)
