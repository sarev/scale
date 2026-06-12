#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The verification floor that every generated doc and comment must pass before it reaches a file - SCALE's local quality
gate. The `Verifier` dataclass combines a deterministic grounding gate with clean-context challenge turns: the gate is
model-free (every backticked word in a doc must appear somewhere in the code corpus), while each challenge runs as a
fresh single-turn conversation so the verdict cannot be biased by the generation context.

Three challenges cover the main failure modes: grounding (name a claim the code does not support), obviousness (does a
block comment add anything beyond the code it sits above), and story (does a doc explain rather than restate). All
fail open - only an explicit negative verdict rejects - so a garbled reply never discards work.

`verify_def` orchestrates the floor for docstrings, running the gate and the grounding challenge in order and granting
at most one feedback-driven regeneration per failed check before the doc is rejected rather than looped on.
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
    Substitute `{name}` placeholders in a template with the given field values.

    Uses plain sequential string replacement rather than `str.format`, so literal braces elsewhere in the template are harmless.

    Parameters:
    - `template`: The template text containing `{name}` placeholders.
    - `fields`: Placeholder names mapped to their replacement strings.

    Returns:
    - The template with every named placeholder replaced.
    """

    # Plain sequential replace rather than str.format, so literal braces elsewhere in the template never raise.
    out = template
    for name, value in fields.items():
        out = out.replace("{" + name + "}", value)
    return out


def ungrounded_tokens(text: str, corpus: str) -> List[str]:
    """
    Collect backticked words in `text` that appear nowhere in `corpus`.

    Only words inside backtick spans are checked, and membership is plain substring containment, so the gate is lenient: any appearance anywhere in the corpus counts as grounded. One-character words are ignored and duplicates are dropped.

    Parameters:
    - `text`: The generated documentation text to check.
    - `corpus`: The source text the identifiers must appear in.

    Returns:
    - The ungrounded words in first-seen order, empty when everything is grounded.
    """

    out: List[str] = []

    # Only words inside backtick spans are checked, and membership is plain substring containment - deliberately lenient, so any appearance anywhere in the corpus counts as grounded.
    for span in _BACKTICK_RE.findall(text or ""):
        for word in _WORD_RE.findall(span):
            if len(word) < 2 or word in out:
                continue
            if word not in corpus:
                out.append(word)

    return out


def _first_word_verdict(reply: str, *words: str) -> Optional[str]:
    """
    Extract a verdict keyword from the first non-empty line of a model reply.

    Only the first line is inspected, so trailing discussion cannot change the verdict; when several keywords appear, the first one listed in `words` wins.

    Parameters:
    - `reply`: The raw model reply, possibly empty.
    - `words`: Candidate verdict keywords, in precedence order.

    Returns:
    - The matched keyword, or `None` when no candidate appears.
    """

    # Judge only the first non-empty line, so trailing chatter after the verdict cannot flip it.
    first = next((ln.strip() for ln in (reply or "").split("\n") if ln.strip()), "")
    upper = first.upper()

    # Caller order is precedence: the first listed word found wins when the line mentions several.
    for w in words:
        if re.search(rf"\b{w}\b", upper):
            return w

    return None


@dataclass
class Verifier:

    """
    Local quality floor for generated documentation: a deterministic grounding gate plus clean-context challenge turns.

    The gate is model-free - backticked words must appear somewhere in `corpus` - while each challenge runs as a fresh single-turn conversation at a fixed temperature so the verdict is not biased by the generation context. The obvious and story challenges fail open: only an explicit negative verdict rejects. Every prompt/feedback field is an optional override of the corresponding built-in template; leave it as `None` to use the default.
    """

    # Every prompt/feedback field is an optional override; None falls back to the built-in template constants.
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


    def ungrounded(self, text: str) -> List[str]:
        """
        List the backticked words in `text` that are absent from this verifier's corpus.

        Parameters:
        - `text`: The documentation text to check.

        Returns:
        - The ungrounded words in first-seen order, empty when everything is grounded.
        """

        return ungrounded_tokens(text, self.corpus)

    # Builds the nudge listing the offending backticked tokens, fed back for one regeneration attempt.
    def gate_feedback(self, tokens: List[str]) -> str:
        """
        Build the nudge message sent back to the writer when the grounding gate finds ungrounded tokens.

        Parameters:
        - `tokens`: The backticked identifiers from the doc that were not found in the code.

        Returns:
        - The nudge text, from the configured template (or the built-in default) with the offending tokens listed.
        """

        return _fill_template(self.gate_nudge or GATE_NUDGE, tokens=", ".join(f"`{t}`" for t in tokens))


    # Every challenge runs as a fresh single-turn conversation at its own temperature, deliberately isolated from the generation context.
    def _ask(self, prompt: str, max_tokens: int) -> str:
        """
        Run a single clean-context challenge turn against the local model.

        The prompt is sent as a lone user message, so the verdict cannot be swayed by the writer conversation's history.

        Parameters:
        - `prompt`: The fully rendered challenge prompt.
        - `max_tokens`: Cap on the reply length, further bounded by the run config's own limit.

        Returns:
        - The stripped reply text, or an empty string if generation produced nothing.
        """

        # Challenge turns run at their own temperature and a tight token cap, in a fresh single-message context.
        turn_cfg = replace(self.cfg, temperature=CHALLENGE_TEMPERATURE,
                           max_new_tokens=min(self.cfg.max_new_tokens, max_tokens))
        return (self.llm.generate([{"role": "user", "content": prompt}], cfg=turn_cfg) or "").strip()

    # An empty or NONE-led reply means the doc passed; anything else is the verdict text, handed back to drive a regeneration.
    def challenge_grounding(self, code: str, doc: str) -> Optional[str]:
        """
        Challenge whether a routine doc's claims are actually grounded in its code.

        A clean-context turn shows the model only the code and the candidate doc and asks it to name an ungrounded claim, or reply `NONE`. An empty or `NONE`-leading reply counts as a pass, so a garbled verdict never raises a complaint.

        Parameters:
        - `code`: The routine's source text.
        - `doc`: The candidate docstring text.

        Returns:
        - The complaint text if the challenge found an ungrounded claim, otherwise `None`.
        """

        # Fail open: an empty or `NONE` reply means no objection, so only a substantive complaint is surfaced.
        prompt = _fill_template(self.grounding_prompt or GROUNDING_PROMPT, code=code, doc=doc)
        reply = self._ask(prompt, GROUNDING_REPLY_TOKENS)
        if not reply or _first_word_verdict(reply, "NONE") == "NONE":
            return None
        return reply

    def grounding_feedback_for(self, verdict: str) -> str:
        """
        Build the feedback message sent back to the writer when the grounding challenge fails.

        Parameters:
        - `verdict`: The challenge turn's complaint describing what the doc got wrong.

        Returns:
        - The feedback text, from the configured template (or the built-in default) with the verdict substituted in.
        """

        return _fill_template(self.grounding_feedback or GROUNDING_FEEDBACK, verdict=verdict)

    # Fail-open: only an explicit NO condemns the comment as obvious, so a garbled reply keeps it.
    def challenge_obvious(self, block: str, comment: str) -> bool:
        """
        Challenge whether a block comment adds anything beyond the code it sits above.

        A clean-context turn shows the model only the block and the candidate comment and asks for a one-word YES/NO verdict. Anything other than an explicit `NO` passes, so a garbled reply never discards a comment.

        Parameters:
        - `block`: The source block the comment describes.
        - `comment`: The candidate comment text.

        Returns:
        - `True` if the comment survives the challenge, `False` on an explicit `NO` verdict.
        """

        prompt = _fill_template(self.obvious_prompt or OBVIOUS_PROMPT, block=block, comment=comment)
        verdict = _first_word_verdict(self._ask(prompt, VERDICT_REPLY_TOKENS), "YES", "NO")

        # Fail open: only an explicit NO discards the comment, so a garbled verdict keeps it.
        return verdict != "NO"

    def obvious_feedback_for(self) -> str:
        """
        Return the feedback text used when a comment fails the obviousness challenge.

        Returns:
        - The configured override, or the built-in default feedback.
        """

        return self.obvious_feedback or OBVIOUS_FEEDBACK

    # Fail-open again: only an explicit RESTATE verdict rejects the doc as a mere restatement of the signature.
    def challenge_story(self, signature: str, doc: str, notes: List[str]) -> bool:
        """
        Challenge whether a routine doc tells a genuine story rather than restating its code.

        A clean-context turn presents the signature, the doc, and the supporting notes, and asks for a one-word STORY/RESTATE verdict. Anything other than an explicit `RESTATE` passes, so an unclear reply never discards a doc.

        Parameters:
        - `signature`: The routine's signature line.
        - `doc`: The candidate docstring text.
        - `notes`: Supporting context notes, rendered as a bullet list.

        Returns:
        - `True` if the doc reads as a story, `False` on an explicit `RESTATE` verdict.
        """

        # Notes render as a bullet list, with an explicit "(none)" placeholder so the template never gets a blank slot.
        listed = "\n".join(f"- {n}" for n in notes) or "(none)"
        prompt = _fill_template(self.story_prompt or STORY_PROMPT, signature=signature, doc=doc, notes=listed)
        verdict = _first_word_verdict(self._ask(prompt, VERDICT_REPLY_TOKENS), "STORY", "RESTATE")

        # Fail open: only an explicit RESTATE rejects the doc.
        return verdict != "RESTATE"

    def story_feedback_for(self) -> str:
        """
        Return the feedback text used when a doc fails the story challenge.

        Returns:
        - The configured override, or the built-in default feedback.
        """

        return self.story_feedback or STORY_FEEDBACK


    # The full def-pass floor: gate ungrounded identifiers with one nudge, then run the grounding challenge with one regeneration; a False result tells the caller to warn rather than ship an unverified doc.
    def verify_def(self, code: str, doc: str, regenerate, label: str = "") -> tuple:
        """
        Run a candidate docstring through the verification floor, repairing it once where possible.

        Two checks run in order: the deterministic grounding gate (every backticked identifier must appear in the code) and the clean-context grounding challenge (a fresh model turn hunting for claims the code does not support). Each failure earns at most one regeneration via `regenerate`; a repeat failure rejects the doc rather than looping.

        Parameters:
        - `code`: The routine's source, the ground truth for both checks.
        - `doc`: The candidate docstring text.
        - `regenerate`: Callback taking a feedback string and returning a replacement doc, or a falsy value to keep the current one.
        - `label`: Routine name used in progress messages.

        Returns:
        - A `(doc, ok)` tuple: the possibly regenerated doc, and `True` if it passed or `False` if it should be dropped.
        """

        # Cheap deterministic gate first: backticked identifiers that never appear in the code.
        tokens = self.ungrounded(doc)

        # Gate failure earns exactly one repair attempt: feed the offending tokens back, then re-check.
        if tokens:
            echo(f"[verify] '{label}': ungrounded identifiers {tokens}; nudging once")
            new = regenerate(self.gate_feedback(tokens))
            if new:
                doc = new
            tokens = self.ungrounded(doc)

            if tokens:
                # A second gate failure rejects the doc outright rather than looping on nudges.
                echo(f"[verify] '{label}': still ungrounded after the nudge ({tokens})")
                return doc, False

        # Stronger check: a clean-context challenge turn hunting for claims the code does not support.
        verdict = self.challenge_grounding(code, doc)

        # On failure, hand the challenger's verdict back as feedback for one regeneration.
        if verdict:
            echo(f"[verify] '{label}': grounding challenge failed; regenerating with the verdict:\n{verdict}")
            new = regenerate(self.grounding_feedback_for(verdict))

            if new:
                doc = new

                # Regeneration can itself smuggle in ungrounded names, so the cheap gate runs again.
                if self.ungrounded(doc):
                    echo(f"[verify] '{label}': regeneration introduced ungrounded identifiers")
                    return doc, False

            # Re-challenge the repaired doc; this is the last chance it gets.
            verdict = self.challenge_grounding(code, doc)

            if verdict:
                # Two challenge failures end the attempt: drop the doc rather than retry indefinitely.
                echo(f"[verify] '{label}': grounding challenge failed twice")
                return doc, False

        # Both checks passed (possibly after repair): the doc is accepted.
        return doc, True
