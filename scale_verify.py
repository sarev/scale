#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

Grounding gate and challenge turns: the local quality floor for generated comments.

A small local model's worst failures are confident inventions - an identifier that does not exist, a behavioural claim
the code does not make, a "comment" that merely restates the line below it. This module checks a candidate comment
*after* it is generated, with two mechanisms that deliberately avoid growing the generation prompt (long rule-laden
prompts make small models drift; a second small, single-aspect turn in a clean context does not):

- **The backtick-grounding gate** (`ungrounded_tokens`) is deterministic and model-free. House style backticks
  identifiers, so every backticked token in a generated doc/comment must appear somewhere in the run's source files
  (a plain substring check against the retained run-file store's text - convention-free, no assumptions about casing
  or naming). A miss earns one nudge naming the offending tokens; a second miss routes to the failure policy.

- **Challenge turns** are small, single-aspect verification questions asked in a CLEAN context - a fresh message list
  with no priming, no file summary, no priors: just the code, the candidate text, and one question, at temperature 0
  with a constrained reply (NONE / YES-NO / STORY-RESTATE) so parsing stays robust for a weak model. Three challenges:
  the **grounding challenge** (def-pass docs: list anything the comment claims that the code does not do), the
  **obviousness challenge** (block comments: does this tell the reader anything not evident from the code?), and the
  **story challenge** (a routine's block comments as a set: do they tell the routine's story or just restate it?).

The shared failure routing lives with the call sites: a challenge failure regenerates once with the verdict as
feedback; a second failure promotes the routine to the escalation manifest when one is active (discarding the local
attempt), else drops a block comment (wrongness is worse than absence) or writes a def doc under a prominent warning
(so the doubt is visible rather than silent). The `Verifier` here only detects and phrases; it never patches - the
byte-for-byte code guarantee is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from scale_log import echo
from typing import List, Optional
import re


# Reply caps (tokens) for the challenge turns. The grounding challenge may list a few false claims; the obviousness
# and story challenges answer in one word, so their caps are tiny.
GROUNDING_REPLY_TOKENS = 160
VERDICT_REPLY_TOKENS = 8

# Challenge turns are decisions, so they run fully deterministic.
CHALLENGE_TEMPERATURE = 0.0


# Built-in prompt defaults for the gate nudge and the three challenge turns. As with the block-pass prompts these are
# fallbacks; the CLI overrides them with the user-editable scale-cfg files `verify.gate.txt`, `verify.grounding.txt`,
# `verify.obvious.txt` and `verify.story.txt`. Placeholders are filled by literal substitution (`_fill_template`), so
# code braces in the substituted text are safe. Each challenge runs in a clean context - the prompt is the entire
# conversation - and asks exactly one question with a constrained reply.
GATE_NUDGE = (
    "Your comment names identifiers that do not exist anywhere in this project's source: {tokens}. Remove or correct "
    "them - never invent an identifier - and give the corrected comment, keeping everything that was accurate."
)
GROUNDING_PROMPT = (
    "Here is a routine, and a documentation comment written for it.\n\n"
    "The code:\n\n{code}\n\n"
    "The comment:\n\n{doc}\n\n"
    "List anything this comment claims that the code shown does not do. Judge only against the code shown. "
    "If the comment claims nothing the code does not do, reply with the single word NONE. Reply with NONE or the "
    "list only - no other commentary."
)
GROUNDING_FEEDBACK = (
    "A reviewer checked your comment against the code and found claims the code does not support:\n\n{verdict}\n\n"
    "Rewrite the comment without those claims - describe only what the code shown actually does - and give the "
    "corrected comment."
)
OBVIOUS_PROMPT = (
    "Here is a paragraph of code, and a comment written to sit above it.\n\n"
    "The code:\n\n{block}\n\n"
    "The comment: {comment}\n\n"
    "Does the comment tell the reader anything that is not already evident from reading the code itself? "
    "Reply with the single word YES or NO."
)
OBVIOUS_FEEDBACK = (
    "A reviewer judged that note as adding nothing a reader of the code would not already see. Give the paragraph's "
    "POINT instead - the reason it is here, a gotcha, or what it accomplishes for the routine - in one short line. "
    "Bare sentence. No #, no quotes, no list."
)
STORY_PROMPT = (
    "Here is a routine's signature, what it does, and the comments written above the paragraphs of its body, "
    "in order.\n\n"
    "The signature:\n\n{signature}\n\n"
    "It does: {doc}\n\n"
    "The paragraph comments:\n{notes}\n\n"
    "Taken together, do these comments tell the story of how the routine achieves what it does, or do they merely "
    "restate the code and pad it with boilerplate? Reply with the single word STORY or RESTATE."
)
STORY_FEEDBACK = (
    "A reviewer judged the notes written for this routine's paragraphs as restating the code rather than telling its "
    "story. For this paragraph, capture WHY it is here and what it contributes to the routine - intent, a reason, a "
    "gotcha - not what the lines plainly say."
)


# A backticked span in a generated comment, and the identifier-shaped words inside one. House style backticks
# identifiers; the gate checks each such word against the run's source text.
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _fill_template(template: str, **fields: str) -> str:
    """
    Substitute `{name}` placeholders by literal replacement (other braces - e.g. in code - are left untouched).

    Parameters:
    - `template`: The prompt template text.
    - `fields`: Placeholder name to replacement value.

    Returns:
    - The filled template.
    """

    out = template
    for name, value in fields.items():
        out = out.replace("{" + name + "}", value)
    return out


def ungrounded_tokens(text: str, corpus: str) -> List[str]:
    """
    Return the backticked identifier words in `text` that appear nowhere in the run's source (the grounding gate).

    Every `backticked` span is scanned for identifier-shaped words; each word of two or more characters must occur as
    a substring of `corpus` (the concatenated source text of every run file). The check is deliberately convention-free
    - no casing or naming assumptions - and case-sensitive, since identifiers are. Single-character words are skipped
    (they occur everywhere, so the check would be vacuous). A word that appears only inside the comment being checked
    obviously cannot ground itself; the corpus is the *source* text, so an invented `ERR_READ_ERROR` is caught even
    when a real `ERR_READ` exists.

    Parameters:
    - `text`: The generated doc/comment text to check.
    - `corpus`: The concatenated source text of the run's files.

    Returns:
    - The ungrounded words, in first-seen order, deduplicated (empty when the text is fully grounded).
    """

    out: List[str] = []
    for span in _BACKTICK_RE.findall(text or ""):
        for word in _WORD_RE.findall(span):
            if len(word) < 2 or word in out:
                continue
            if word not in corpus:
                out.append(word)
    return out


def _first_word_verdict(reply: str, *words: str) -> Optional[str]:
    """
    Find which of `words` a constrained challenge reply leads with, scanning its first non-empty line.

    Parameters:
    - `reply`: The raw model reply.
    - `words`: The candidate verdict words (upper-case).

    Returns:
    - The matched word, or None when the reply names none of them.
    """

    first = next((ln.strip() for ln in (reply or "").split("\n") if ln.strip()), "")
    upper = first.upper()
    for w in words:
        if re.search(rf"\b{w}\b", upper):
            return w
    return None


@dataclass
class Verifier:
    """
    The run's verification harness: the deterministic grounding gate plus the clean-context challenge turns.

    One instance is built per run (over the retained run-file store's concatenated source text) and threaded into the
    definition and block passes. Every challenge runs against a FRESH message list - no priming, no file summary, no
    priors - so the judging model cannot be led by the generation context that produced the candidate text. The
    verifier only detects problems and phrases feedback; the regenerate/promote/drop routing is the call sites'.

    Attributes:
    - `llm`/`cfg`: The model and base generation configuration (challenges clone `cfg` at temperature 0).
    - `corpus`: The concatenated source text of every run file (targets and references) for the grounding gate.
    - `gate_nudge`/`grounding_prompt`/`grounding_feedback`/`obvious_prompt`/`obvious_feedback`/`story_prompt`/
      `story_feedback`: Optional prompt-wording overrides (the scale-cfg files); None uses the built-in defaults.
    """

    llm: object
    cfg: object
    corpus: str = ""
    gate_nudge: Optional[str] = None
    grounding_prompt: Optional[str] = None
    grounding_feedback: Optional[str] = None
    obvious_prompt: Optional[str] = None
    obvious_feedback: Optional[str] = None
    story_prompt: Optional[str] = None
    story_feedback: Optional[str] = None

    # ---------------------------- the deterministic gate ----------------------------

    def ungrounded(self, text: str) -> List[str]:
        """Return the backticked identifier words in `text` that the run's source nowhere contains."""
        return ungrounded_tokens(text, self.corpus)

    def gate_feedback(self, tokens: List[str]) -> str:
        """Phrase the one corrective nudge for a gate failure, naming the offending tokens."""
        return _fill_template(self.gate_nudge or GATE_NUDGE, tokens=", ".join(f"`{t}`" for t in tokens))

    # ---------------------------- the clean-context challenges ----------------------------

    def _ask(self, prompt: str, max_tokens: int) -> str:
        """Ask one challenge question in a clean context (a fresh, single-turn message list) at temperature 0."""
        turn_cfg = replace(self.cfg, temperature=CHALLENGE_TEMPERATURE,
                           max_new_tokens=min(self.cfg.max_new_tokens, max_tokens))
        return (self.llm.generate([{"role": "user", "content": prompt}], cfg=turn_cfg) or "").strip()

    def challenge_grounding(self, code: str, doc: str) -> Optional[str]:
        """
        Ask, in a clean context, what the candidate doc claims that the code does not do (the grounding challenge).

        Parameters:
        - `code`: The routine snippet the doc was written for (the model's elided view is fine).
        - `doc`: The candidate documentation text.

        Returns:
        - None when the challenge passes (the reply is NONE), else the verdict text (the listed false claims).
        """

        prompt = _fill_template(self.grounding_prompt or GROUNDING_PROMPT, code=code, doc=doc)
        reply = self._ask(prompt, GROUNDING_REPLY_TOKENS)
        if not reply or _first_word_verdict(reply, "NONE") == "NONE":
            return None
        return reply

    def grounding_feedback_for(self, verdict: str) -> str:
        """Phrase the regeneration feedback for a grounding-challenge failure, carrying the verdict."""
        return _fill_template(self.grounding_feedback or GROUNDING_FEEDBACK, verdict=verdict)

    def challenge_obvious(self, block: str, comment: str) -> bool:
        """
        Ask, in a clean context, whether a block comment tells the reader anything beyond the code (obviousness).

        Parameters:
        - `block`: The paragraph's pristine source text.
        - `comment`: The candidate one-line comment.

        Returns:
        - True when the comment passes (YES, or an unparseable reply - the benefit of the doubt); False on a NO.
        """

        prompt = _fill_template(self.obvious_prompt or OBVIOUS_PROMPT, block=block, comment=comment)
        verdict = _first_word_verdict(self._ask(prompt, VERDICT_REPLY_TOKENS), "YES", "NO")
        return verdict != "NO"

    def obvious_feedback_for(self) -> str:
        """Phrase the regeneration feedback for an obviousness failure."""
        return self.obvious_feedback or OBVIOUS_FEEDBACK

    def challenge_story(self, signature: str, doc: str, notes: List[str]) -> bool:
        """
        Ask, in a clean context, whether a routine's paragraph notes tell its story or merely restate it.

        Parameters:
        - `signature`: The routine's header text.
        - `doc`: The routine's one-line doc summary.
        - `notes`: The full set of turn-1 paragraph summaries, in body order.

        Returns:
        - True when the set passes (STORY, or an unparseable reply); False on a RESTATE verdict.
        """

        listed = "\n".join(f"- {n}" for n in notes) or "(none)"
        prompt = _fill_template(self.story_prompt or STORY_PROMPT, signature=signature, doc=doc, notes=listed)
        verdict = _first_word_verdict(self._ask(prompt, VERDICT_REPLY_TOKENS), "STORY", "RESTATE")
        return verdict != "RESTATE"

    def story_feedback_for(self) -> str:
        """Phrase the per-paragraph regeneration feedback for a story-challenge failure."""
        return self.story_feedback or STORY_FEEDBACK

    # ---------------------------- def-pass orchestration ----------------------------

    def verify_def(self, code: str, doc: str, regenerate, label: str = "") -> tuple:
        """
        Run the def-pass verification pipeline on a candidate docstring/header comment: gate, then grounding challenge.

        Each mechanism allows one corrective regeneration (the gate's nudge names the offending tokens; the
        challenge's feedback carries the verdict), produced by the caller's `regenerate(feedback)` - which reopens the
        worker's own generation context so the model sees its snippet, its previous answer, and the feedback. A
        regeneration that yields nothing usable (the callback returns "") keeps the previous candidate. A second
        failure of either mechanism reports `ok=False`; the routing - promote to the manifest, or write under a
        prominent warning - is the caller's.

        Parameters:
        - `code`: The snippet the doc was generated from (used by the grounding challenge).
        - `doc`: The candidate documentation text.
        - `regenerate`: Callback `feedback -> new_doc_or_empty` regenerating in the worker's context.
        - `label`: The routine's qualname, for logging.

        Returns:
        - `(doc, ok)`: the (possibly regenerated) documentation text, and whether it passed verification.
        """

        # The deterministic gate first (free): backticked identifiers must exist in the run's source.
        tokens = self.ungrounded(doc)
        if tokens:
            echo(f"[verify] '{label}': ungrounded identifiers {tokens}; nudging once")
            new = regenerate(self.gate_feedback(tokens))
            if new:
                doc = new
            tokens = self.ungrounded(doc)
            if tokens:
                echo(f"[verify] '{label}': still ungrounded after the nudge ({tokens})")
                return doc, False

        # Then the grounding challenge (one clean-context model turn): claims must be supported by the code.
        verdict = self.challenge_grounding(code, doc)
        if verdict:
            echo(f"[verify] '{label}': grounding challenge failed; regenerating with the verdict:\n{verdict}")
            new = regenerate(self.grounding_feedback_for(verdict))
            if new:
                doc = new
                # A regeneration may itself invent an identifier; the gate is free, so re-check it.
                if self.ungrounded(doc):
                    echo(f"[verify] '{label}': regeneration introduced ungrounded identifiers")
                    return doc, False
            verdict = self.challenge_grounding(code, doc)
            if verdict:
                echo(f"[verify] '{label}': grounding challenge failed twice")
                return doc, False

        return doc, True
