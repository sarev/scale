#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

Shared text-shaping utilities for fitting prose and source into the model's context window. `summarise` is the general
summarising turn used throughout the tool: a length keyword selects both the instruction wording and a reply-token
cap, either of which the caller may override explicitly, and the cap only ever tightens the configured budget.

The other two functions crop code rather than prose: `elide_to_budget` trims a snippet's body to a token budget by
keeping its head and tail around an elision marker, always preserving the header lines verbatim, and `fit_snippet`
measures the budget actually left beside the current conversation - scaled down by a safety factor so estimation error
cannot overflow the window - before applying the same elision.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Optional, Tuple


# Length presets for `summarise`. Each names a target shape and carries a reply-token cap sized to it, so a one-line
# summary cannot ramble into a paragraph and a file overview is not truncated mid-sentence. Callers may override the cap
# (e.g. the whole-file summary uses its own larger budget).
LENGTH_LINE = "one line"
LENGTH_PARAGRAPH = "one paragraph"
LENGTH_PARAGRAPHS = "a few paragraphs"


# Fixed acknowledgement the priming appends after each context turn (instead of asking the model to generate "OK",
# which conditions a small model to answer the first real request with "OK" too). Lives here, the lowest-level shared
# module, so both the orchestrator (which emits it) and the Python worker (which must recognise a parroted echo of it as
# a non-answer) can reference the one string.
PRIMING_ACK = "Understood - ready for the code."

_LENGTH_RESERVE = {
    LENGTH_LINE: 64,
    LENGTH_PARAGRAPH: 200,
    LENGTH_PARAGRAPHS: 512,
}

_LENGTH_INSTRUCTION = {
    LENGTH_LINE: "Summarise it in a single short line - just the line, no preamble.",
    LENGTH_PARAGRAPH: "Summarise it briefly in one paragraph.",
    LENGTH_PARAGRAPHS: "Summarise it in a few short paragraphs, noting any significant internal details.",
}


def summarise(
    llm,
    cfg,
    text: str,
    length: str = LENGTH_PARAGRAPH,
    *,
    base_messages: Optional[List[dict]] = None,
    subject: str = "the following",
    max_tokens: Optional[int] = None,
    instruction: Optional[str] = None,
) -> str:
    """
    Ask the model for a summary of arbitrary text at a chosen length.

    The `length` keyword selects both the instruction wording and a reply-token cap from internal tables; an explicit `instruction` or `max_tokens` overrides the corresponding table entry. The cap only ever tightens the configured token budget, never raises it.

    Parameters:
    - `llm`: The chat model used to generate the summary.
    - `cfg`: The generation configuration; its per-turn token budget is clamped to the cap.
    - `text`: The text to be summarised.
    - `length`: Target length key selecting the instruction and token cap.
    - `base_messages`: Optional conversation prefix to prepend before the summary request.
    - `subject`: Phrase naming what the text is, woven into the prompt.
    - `max_tokens`: Optional explicit reply-token cap, overriding the length default.
    - `instruction`: Optional explicit summarising instruction, overriding the length default.

    Returns:
    - The model's summary with surrounding whitespace stripped.
    """

    # Both the instruction wording and the reply-token cap come from the per-length tables; the cap only ever tightens cfg.max_new_tokens, never raises it.
    if instruction is None:
        instruction = _LENGTH_INSTRUCTION.get(length, _LENGTH_INSTRUCTION[LENGTH_PARAGRAPH])
    prompt = (
        f"Here is {subject}:\n\n{text}\n\n"
        f"{instruction} Do not ask questions or add any conversational discussion; give only the summary."
    )
    cap = max_tokens if max_tokens is not None else _LENGTH_RESERVE.get(length, _LENGTH_RESERVE[LENGTH_PARAGRAPH])
    turn_cfg = replace(cfg, max_new_tokens=min(cfg.max_new_tokens, cap))
    messages = (base_messages or []) + [{"role": "user", "content": prompt}]

    return llm.generate(messages, cfg=turn_cfg).strip()


# Per-language markers inserted where a routine body has been elided. Each is a `str.format` template taking `n`
# (the number of omitted lines) and is rendered in the language's own comment style, so the snippet still reads as
# plausible source to the model. The marker is only ever shown to the LLM; it is never written to the output file.
MARKER_PYTHON = "# ... {n} lines omitted for brevity ..."
MARKER_JS = "// ... {n} lines omitted for brevity ..."
MARKER_C = "/* ... {n} lines omitted for brevity ... */"


def elide_to_budget(
    snippet_lines: List[str],
    header_line_count: int,
    budget_tokens: int,
    estimate_fn: Callable[[str], int],
    marker: str,
    head_tail_ratio: float = 0.5,
) -> Tuple[List[str], int]:
    """
    Trim a snippet's body to a token budget by keeping its head and tail around an elision marker.

    The header lines always survive verbatim; the header and the marker are charged against the budget first, and the remaining body budget is split between head and tail by `head_tail_ratio`. If the snippet already fits, or nothing would actually be dropped, the input lines are returned unchanged.

    Parameters:
    - `snippet_lines`: The snippet as a list of lines.
    - `header_line_count`: Number of leading lines that must always be kept.
    - `budget_tokens`: Total token budget for the result.
    - `estimate_fn`: Callable estimating the token count of a string.
    - `marker`: Elision marker template; `{n}` is replaced with the omitted line count.
    - `head_tail_ratio`: Fraction of the body budget given to the head.

    Returns:
    - A tuple of the (possibly elided) lines and the number of omitted lines, 0 when nothing was dropped.
    """

    # Fast path when everything already fits; otherwise the header and the marker itself are charged first, and only the remainder is split between head and tail by the ratio.
    if estimate_fn("\n".join(snippet_lines)) <= budget_tokens:
        return snippet_lines, 0
    header = snippet_lines[:header_line_count]
    body = snippet_lines[header_line_count:]
    if not body:
        return snippet_lines, 0
    header_tokens = estimate_fn("\n".join(header)) if header else 0
    marker_tokens = estimate_fn(marker.format(n=len(body)))
    body_budget = budget_tokens - header_tokens - marker_tokens
    if body_budget <= 0:
        return header + [marker.format(n=len(body))], len(body)
    head_budget = int(body_budget * head_tail_ratio)
    tail_budget = body_budget - head_budget
    head: List[str] = []
    used = 0
    i = 0

    # Grow the head greedily from the top, charging an extra token per line to approximate the joining newline.
    while i < len(body):
        cost = estimate_fn(body[i]) + 1  # +1 approximates the joining newline
        if used + cost > head_budget:
            break
        head.append(body[i])
        used += cost
        i += 1

    # Now fill the tail backwards from the end under its own budget.
    tail: List[str] = []
    used_tail = 0
    j = len(body) - 1

    # The j >= i bound stops the tail before it re-includes any line the head already kept.
    while j >= i:
        cost = estimate_fn(body[j]) + 1
        if used_tail + cost > tail_budget:
            break
        tail.insert(0, body[j])
        used_tail += cost
        j -= 1

    # If the head and tail met, nothing was dropped: return the original rather than splicing in a pointless marker.
    omitted = len(body) - len(head) - len(tail)
    if omitted <= 0:
        return snippet_lines, 0
    return header + head + [marker.format(n=omitted)] + tail, omitted


def fit_snippet(
    llm,
    cfg,
    messages,
    snippet: str,
    header_line_count: int,
    marker: str,
    head_tail_ratio: float = 0.5,
    safety: float = 0.9,
) -> Tuple[str, int]:
    """
    Crop a snippet to the model's remaining context budget, eliding the middle of its body.

    The budget is measured against the current conversation and scaled down by `safety` so token-estimation error cannot overflow the context window.

    Parameters:
    - `llm`: The chat model, used for the budget measurement and token estimates.
    - `cfg`: The generation configuration.
    - `messages`: The conversation the snippet must fit alongside.
    - `snippet`: The snippet text to crop.
    - `header_line_count`: Number of leading lines that must always be kept.
    - `marker`: Elision marker template; `{n}` is replaced with the omitted line count.
    - `head_tail_ratio`: Fraction of the body budget given to the head.
    - `safety`: Fraction of the measured budget actually used.

    Returns:
    - A tuple of the (possibly elided) snippet text and the number of omitted lines, 0 when it already fits.
    """

    # The safety factor deliberately undershoots the measured budget so token-estimate error cannot overflow the context window.
    budget = int(llm.snippet_budget(messages, cfg) * safety)
    lines = snippet.split("\n")
    elided, omitted = elide_to_budget(lines, header_line_count, budget, llm.estimate_tokens, marker, head_tail_ratio)
    if omitted == 0:
        return snippet, 0
    return "\n".join(elided), omitted
