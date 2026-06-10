#!/usr/bin/env python3
"""
The file-doc safety veto: `looks_legal` must recognise license/legal boilerplate so the engine never rewrites it,
even if the model misclassifies a legal line as the file description. Over-matching is acceptable (we just decline to
touch that line); under-matching is the dangerous failure, so this pins the markers we rely on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_filedoc import looks_legal  # noqa: E402

LEGAL = [
    "Copyright 2025 7th software Ltd.",
    "Copyright (c) 2024 Acme Corp",
    "SPDX-License-Identifier: Apache-2.0",
    "Licensed under the Apache License, Version 2.0",
    "Permission is hereby granted, free of charge,",
    "Redistribution and use in source and binary forms",
    "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND",
    "All rights reserved.",
    "Released under the MIT License",
    "GNU General Public License",
    "© 2025 Someone",
]

PROSE = [
    "Implements the workspace allocator used by the editor core.",
    "Parses incoming requests and dispatches them to the right handler.",
    "Small helpers for formatting timestamps and durations.",
    "Entry point: wires up the subsystems and runs the main loop.",
]


def main():
    for line in LEGAL:
        assert looks_legal(line), f"should be flagged as legal: {line!r}"
    for line in PROSE:
        assert not looks_legal(line), f"plain description wrongly flagged: {line!r}"
    print("PASS: looks_legal flags license/legal text and leaves plain descriptions editable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
