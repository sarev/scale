#!/usr/bin/env python3
"""
Run every test_*.py in this directory and report a pass/fail summary.

These tests are fast and require NO GGUF model: they stub the LLM or exercise
only the parsing / patching / caching logic. Run them with the project venv:

    python tests/run_all.py

A non-zero exit code indicates at least one failing test.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    tests = sorted(HERE.glob("test_*.py"))
    failures = []
    for t in tests:
        print(f"\n===== {t.name} =====")
        if subprocess.run([sys.executable, str(t)]).returncode != 0:
            failures.append(t.name)

    print("\n" + "=" * 48)
    if failures:
        print(f"FAILED ({len(failures)}/{len(tests)}): {', '.join(failures)}")
        return 1
    print(f"All {len(tests)} test files passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
