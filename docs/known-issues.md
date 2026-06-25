# Known issues and hardening backlog

Issues and friction observed while driving a large **online** Python block-comment run (roughly 500
routines across 40 modules, `-bc`/`--block-comments` only, no def pass). Each entry notes the symptom, where it
bit, a proposed fix, and whether the fix belongs in the tool or in the `scale` skill document.

**All six issues and the driver pitfalls below were addressed 2026-06-25** — see the Resolution section at the
end for the per-issue commits.

Severity key: **high** = silent correctness/quality loss; **medium** = friction or lint breakage that a
careful driver works around; **low** = wording/clarity.

## 1. Existing hand-written comments are silently overwritten (high)

The patcher replaces an existing same-indent comment whenever the comment pass supplies fresh text for that
chunk. On a codebase that is already commented, the pass therefore *degrades* the very blocks that were most
carefully written: in this run it collapsed a four-line rationale into one line, duplicated an existing
comment one line above its original, and overall did enough damage to two well-commented files that they were
reverted wholesale.

The behaviour is documented for the patcher (a `NONE` answer keeps an existing comment; a written answer
replaces it), but nothing steers the *writer* toward `NONE` when a good comment already exists, so the default
online path destroys prior work.

Proposed fix (ideally both):
- **Tool**: do not overwrite a *substantive* existing comment unless explicitly asked. A one-line restate is
  fair game; a multi-line block, or one the value score would rate highly, should be preserved by default
  (perhaps a `--block-comments` mode or a `keep-existing` flag). At minimum, surface the existing comment text
  per chunk in the manifest so the writer can make an informed choice.
- **Skill**: instruct the fill agents explicitly — "if the chunk's snippet already shows an adequate leading
  comment, answer `NONE` to preserve it; only write where a blob has none or a poor one."

## 2. Chunk line-drift in long, already-commented routines (high)

The manifest `snippet` is raw source and each chunk's `lines` is a 1-based inclusive range into it. When the
body already contains comments and blank lines, the line counting is easy to get wrong, and a comment can land
on the wrong block. In this run one comment about durable-metadata persistence was placed above an unrelated
`except` handler; a separate agent noticed the same interleaving in a long function and had to reassign
answers by `bidx` by hand. Silent misplacement is the dangerous case.

Proposed fix:
- **Tool**: render the manifest `snippet` with a line-number gutter (the machinery already exists for the LLM
  segment fallback, `render_numbered_body`), so the writer maps chunks to code by number rather than by
  counting. This removes the whole class of off-by-N errors.

## 3. No line-length budget, and block comments are not wrapped (medium)

Comments are inserted verbatim. A one-line answer that looked short became a >120-column line once indented
inside a method, tripping the project's linter (13 such lines in this run, 121-145 cols). The writer has no
reliable way to anticipate the final indentation.

Proposed fix:
- **Tool**: auto-wrap an inserted block comment to the configured `line-length` (the patcher knows both the
  indent and the limit), continuing with `#` on the next line. This is the robust fix.
- **Skill** (interim): state the budget — indent + `# ` + text must fit the line-length — and suggest keeping
  answer text conservatively short (e.g. under ~90 characters) because indentation is unknown at fill time.

## 4. No project-local configuration (medium)

`scale_path` is hardcoded to the script's own `scale-cfg`, so tuning a language's comment template (here, the
Python doc-comment format to a house style) means editing the tool's *global* default and changing behaviour
for every other project. There is no per-run override.

Proposed fix:
- **Tool**: a `--config-dir` flag, or discovery of a project-local `scale-cfg`/`.scale-cfg` that overlays the
  built-in templates, so a repo can pin its own house style without mutating the shared install.

## 5. The manifest fill method is underspecified (medium, skill)

The skill says to "fill every `answer` field" but gives no reliable mechanism. Hand-editing the `null`s is
fragile because `"answer": null` is not unique, so a naive find/replace is unsafe. A robust recipe that worked
well: have the agent build a flat list of answers in document order (each request in order, its chunks in
array order) and assign them programmatically — load the JSON, iterate `requests[].blocks.chunks[]`, pop one
answer each, assert the list is exhausted, dump. The `--check-manifest` gate then confirms completeness.

Proposed fix:
- **Skill**: document that recipe (or similar) as the recommended fill method.
- **Tool** (optional): a `--fill-fragment <frag> --answers <ordered.json>` helper would remove the need for
  agents to write any JSON at all.

## 6. No guidance for large runs (medium, skill)

The skill's "check out fragments, spawn one fresh subagent per fragment, all in parallel" reads as a single
up-front fan-out. For ~500 routines that implies dozens of simultaneous agents, which is wasteful and hard to
oversee. The mechanism actually supports a far better rhythm that the skill under-sells: check out a batch,
fill it, run `--apply-manifest` to *bank the finished work and report the remainder* (it merges filled
fragments, deletes the spent files, and returns any unfilled slots to the pile), then repeat until the apply
completes and patches the source. The source is only ever patched when the master is fully filled, so applying
between rounds is safe and purely additive.

Proposed fix:
- **Skill**: present the iterative round-based loop as the recommended path for large runs, and clarify that
  `--fragment-size` counts **routines, not comments**, so sizing a batch by "~N comments" needs a rough
  routine-to-chunk ratio (this run averaged ~2.4 chunks per routine).

## Driver pitfalls worth a note (not tool bugs)

These cost time in this run but are operator error, not SCALE faults. A short "driver notes" aside in the
skill could spare the next person:

- **Do not pre-plan the whole fan-out.** Work in rounds (see issue 6); let `--apply-manifest` tell you what is
  left rather than cutting every fragment up front.
- **Encoding when verifying.** Reading `git show HEAD:<file>` without forcing UTF-8 decodes non-ASCII (em
  dashes, ellipses, bullets) as the platform default and fakes a "corruption" diff. SCALE itself preserved all
  non-ASCII correctly; the scare was in the check, not the files. Always compare with an explicit UTF-8 decode.

## Resolution (2026-06-25)

Each issue was fixed in its own commit, with a model-free regression test, suite green throughout.

1. **Existing comments overwritten** — `defer_block_targets` now surfaces an attached comment as the chunk's
   `existing` text and protects a multi-line one by pre-answering `NONE` + `preserve` (apply keeps it verbatim);
   `--overwrite-comments` opts out. Skill steers writers to decline adequate comments.
   (`test_block_preserve_existing.py`.)
2. **Chunk line-drift** — each chunk carries an `anchor` (the verbatim boundary-line text) so the writer matches
   the line instead of counting; the snippet stays verbatim. Chose this over a line-number gutter, which would
   have broken the snippet-slicing contract. (`test_block_chunk_anchor.py`.)
3. **No line-length / no wrapping** — new `--line-length N` wraps inserted block comments at the patcher (where the
   indent is known); default off. Online: stored in the manifest at emit, honoured at apply, CLI-overridable.
   (`test_comment_wrap.py`.)
4. **No project-local config** — `--config-dir`, or an auto-discovered `scale-cfg`/`.scale-cfg`, overlays the
   built-in templates per file via `ConfigResolver`. (`test_config_overlay.py`.)
5. **Underspecified fill method** — the `scale` skill now documents the programmatic fill recipe (iterate
   `requests[].blocks.chunks[]` in order, assign, assert none null) instead of unsafe `null` find/replace.
6. **No large-run guidance** — the skill presents the round-based loop (check out a wave, fill, apply to bank and
   learn the remainder, repeat), recommends ~15–20-agent waves, and clarifies `--fragment-size` counts routines.

Driver pitfalls (don't pre-plan the whole fan-out; force UTF-8 when verifying) were added to the skill's driver
notes. The optional `--fill-fragment --answers` helper was not built (agents fill fragment JSON reliably via the
documented recipe).
