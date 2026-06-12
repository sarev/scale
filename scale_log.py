#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

SCALE's minimal logging layer, shared by every module in the tool. `echo` prints progress messages only when verbose
mode is enabled, forcing a flush so output keeps pace with long model runs, while `error` writes to stderr regardless
of verbosity.

`set_verbosity` flips the module-level switch for the whole process, so the CLI's verbose flag controls progress
output everywhere without logger objects to thread through.
"""

from __future__ import annotations

import sys


VERBOSE = False


def echo(*args, **kwargs):
    """
    Print a progress message when verbose mode is enabled.

    Parameters:
    - `args`: Positional values forwarded to `print`.
    - `kwargs`: Keyword options forwarded to `print`; `flush` is always forced on.
    """

    # Force a flush so progress lines appear promptly even when output is piped.
    if VERBOSE:
        kwargs["flush"] = True
        print(*args, **kwargs)


def error(*args, **kwargs):
    """
    Write an error message to stderr, regardless of verbosity.

    Parameters:
    - `args`: Values joined with spaces to form the message.
    """

    # Bypasses the verbosity gate and flushes immediately so errors are never lost.
    msg = " ".join(str(a) for a in args)
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def set_verbosity(state: bool):
    """
    Enable or disable verbose progress output.

    Parameters:
    - `state`: Set verbose output to enabled (`True`) or disabled (`False`).
    """

    global VERBOSE
    VERBOSE = state
