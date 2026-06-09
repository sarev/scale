#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

Helpers for keeping the text SCALE sends to the LLM within the model's context window.

The expensive, fragile case is a single routine whose body is larger than the context window. Because SCALE patches
comments into the parsed source (and never has the LLM re-emit code), the snippet the model *reads* can be reduced
freely without any risk to the output: the comment will simply be written from a partial view of the body. This module
provides that reduction - it keeps a routine's signature plus the head and tail of its body and elides the middle,
replacing it with a short marker noting how many lines were omitted.
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
) -> str:
    """
    Ask the model to summarise `text` into a target length, and return the summary.

    This is SCALE's one reusable "compress some text" call. It is used wherever the tool needs the model to distil
    something - a whole source file, a chunk of one, or a deeply-nested block being elided to fit the context window -
    so the prompt shape and the reply-length discipline live in a single place. The reply is capped to a size matching
    `length` (overridable via `max_tokens`) so a one-line ask cannot ramble.

    Parameters:
    - `llm`: A model exposing `generate`.
    - `cfg`: The base generation configuration (cloned with a length-appropriate `max_new_tokens`).
    - `text`: The text to summarise.
    - `length`: One of `LENGTH_LINE` / `LENGTH_PARAGRAPH` / `LENGTH_PARAGRAPHS`.
    - `base_messages`: Optional priming context to prepend (e.g. the system prompt + file overview); omit for a cheap,
      standalone summary that needs no wider context.
    - `subject`: A short noun phrase describing what `text` is, woven into the prompt (e.g. "a Python source file").
    - `max_tokens`: Optional reply-token cap overriding the per-length default.

    Returns:
    - The summary text, stripped.
    """

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
    Reduce a routine snippet to fit a token budget by eliding the middle of its body.

    The header (signature, decorators, opening line(s)) is always preserved, since it carries the most intent for the
    least cost. The remaining budget is spent on the start and end of the body - the start typically holds setup and
    guard clauses, the end the return value - with the uninformative middle replaced by `marker`. If the snippet
    already fits, it is returned unchanged.

    Parameters:
    - `snippet_lines`: The full snippet split into lines (header followed by body).
    - `header_line_count`: How many leading lines form the header and must always be kept.
    - `budget_tokens`: The maximum number of tokens the returned snippet may occupy.
    - `estimate_fn`: A cheap function estimating the token count of a string (e.g. `LocalChatModel.estimate_tokens`).
    - `marker`: A `str.format` template taking `n`, inserted where the body was elided.
    - `head_tail_ratio`: The fraction of the body budget spent on the head (the remainder goes to the tail).

    Returns:
    - A tuple `(lines, omitted)` where `lines` is the (possibly reduced) snippet and `omitted` is the number of body
      lines removed (0 if no elision was necessary).

    Notes:
    - The token sizing is approximate (it relies on `estimate_fn`); callers wanting a hard guarantee should pass a
      budget with a safety margin already applied.
    - In the degenerate case where even the header exceeds the budget, the header is returned together with a marker
      reporting that the whole body was omitted.
    """

    if estimate_fn("\n".join(snippet_lines)) <= budget_tokens:
        return snippet_lines, 0

    header = snippet_lines[:header_line_count]
    body = snippet_lines[header_line_count:]
    if not body:
        # Nothing to trim (the header alone is over budget); leave it to the caller's last-resort guard.
        return snippet_lines, 0

    header_tokens = estimate_fn("\n".join(header)) if header else 0
    marker_tokens = estimate_fn(marker.format(n=len(body)))
    body_budget = budget_tokens - header_tokens - marker_tokens
    if body_budget <= 0:
        # No room for any body lines: keep just the header and note the full body was dropped.
        return header + [marker.format(n=len(body))], len(body)

    head_budget = int(body_budget * head_tail_ratio)
    tail_budget = body_budget - head_budget

    # Greedily keep body lines from the top up to the head budget.
    head: List[str] = []
    used = 0
    i = 0
    while i < len(body):
        cost = estimate_fn(body[i]) + 1  # +1 approximates the joining newline
        if used + cost > head_budget:
            break
        head.append(body[i])
        used += cost
        i += 1

    # Greedily keep body lines from the bottom up to the tail budget, without overlapping the head.
    tail: List[str] = []
    used_tail = 0
    j = len(body) - 1
    while j >= i:
        cost = estimate_fn(body[j]) + 1
        if used_tail + cost > tail_budget:
            break
        tail.insert(0, body[j])
        used_tail += cost
        j -= 1

    omitted = len(body) - len(head) - len(tail)
    if omitted <= 0:
        # Everything fit once split head/tail (rare); return the original untouched.
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
    Elide a routine snippet, if necessary, so it fits the context window of an `llm`.

    This is the convenience wrapper the language workers call. It asks the model for the snippet's token budget
    (`llm.snippet_budget`), then applies `elide_to_budget` using the model's cheap token estimator. A `safety` factor
    is applied to the budget to absorb estimation error, since the estimate is approximate.

    Parameters:
    - `llm`: A `LocalChatModel` exposing `snippet_budget` and `estimate_tokens`.
    - `cfg`: The generation configuration (used for the reply reserve in the budget calculation).
    - `messages`: The persistent priming context the snippet will be appended to.
    - `snippet`: The assembled routine snippet (header plus body) as a single string.
    - `header_line_count`: How many leading lines form the header and must always be kept.
    - `marker`: A `str.format` template taking `n`, inserted where the body is elided.
    - `head_tail_ratio`: The fraction of the body budget spent on the head.
    - `safety`: Multiplier (< 1) applied to the budget to allow for token-estimate error.

    Returns:
    - A tuple `(snippet, omitted)` where `snippet` is the (possibly reduced) text and `omitted` is the number of body
      lines removed (0 if it already fit).
    """

    budget = int(llm.snippet_budget(messages, cfg) * safety)
    lines = snippet.split("\n")
    elided, omitted = elide_to_budget(lines, header_line_count, budget, llm.estimate_tokens, marker, head_tail_ratio)
    if omitted == 0:
        return snippet, 0
    return "\n".join(elided), omitted
