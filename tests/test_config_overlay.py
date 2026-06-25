#!/usr/bin/env python3
"""
Project-local prompt overrides overlay the built-in scale-cfg (known-issue #4).

Tuning a language's comment template used to mean editing the tool's global scale-cfg, changing behaviour for every
other project. `ConfigResolver` resolves each `config_dir / "name"` lookup across an ordered list (overrides first,
built-in last), so a repo can pin its own house style by dropping just the templates it wants to change into a
`scale-cfg`/`.scale-cfg` directory; everything else inherits the built-in. `_discover_config_dir` finds that
directory at or above the working directory and never mistakes the tool's own scale-cfg for a project override.
No GGUF model required.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale import ConfigResolver, _discover_config_dir, _read_optional  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        builtin = root / "builtin"
        override = root / "proj" / "scale-cfg"
        builtin.mkdir(parents=True)
        override.mkdir(parents=True)

        # The built-in carries both templates; the override pins only one of them.
        (builtin / "comment.txt").write_text("BUILTIN comment", encoding="utf-8")
        (builtin / "summary.txt").write_text("BUILTIN summary", encoding="utf-8")
        (override / "comment.txt").write_text("OVERRIDE comment", encoding="utf-8")

        # ---- 1. Overlay: override wins where present, built-in fills the rest, unknown falls to built-in path ----
        res = ConfigResolver([override, builtin])
        assert _read_optional(res / "comment.txt") == "OVERRIDE comment", "an overridden template must win"
        assert _read_optional(res / "summary.txt") == "BUILTIN summary", "an un-overridden template must inherit"
        missing = res / "nope.txt"
        assert missing == builtin / "nope.txt" and not missing.is_file(), \
            "an unknown name resolves to the built-in path (absent), as a bare Path did"

        # ---- 2. Built-in only: every lookup resolves there ----
        only = ConfigResolver([builtin])
        assert _read_optional(only / "comment.txt") == "BUILTIN comment"
        assert _read_optional(only / "summary.txt") == "BUILTIN summary"

        # ---- 3. Discovery: finds an override at or above start, excludes the built-in, prefers scale-cfg ----
        start = override.parent / "src" / "pkg"
        start.mkdir(parents=True)
        found = _discover_config_dir(start, builtin)
        assert found is not None and found.resolve() == override.resolve(), \
            f"discovery must walk up to the project scale-cfg, got {found}"

        # A `.scale-cfg` is discovered too.
        dotdir = root / "proj2"
        (dotdir / ".scale-cfg").mkdir(parents=True)
        assert _discover_config_dir(dotdir, builtin).resolve() == (dotdir / ".scale-cfg").resolve()

        # The built-in itself is never returned as an override: running the tool from a root whose own scale-cfg IS
        # the built-in must see no overlay (the candidate dir equals the built-in, so it is skipped).
        home = root / "tool"
        tool_cfg = home / "scale-cfg"
        tool_cfg.mkdir(parents=True)
        assert _discover_config_dir(home, tool_cfg) is None, "the built-in scale-cfg must not count as an override"

        # No config dir anywhere up the tree -> None.
        bare = root / "bare" / "deep"
        bare.mkdir(parents=True)
        assert _discover_config_dir(bare, builtin) is None, "no scale-cfg found means no overlay"

    print("PASS: project-local scale-cfg overrides overlay the built-in per file; discovery excludes the built-in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
