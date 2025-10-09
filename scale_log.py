#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

This module provides a basic logging and verbosity control system for the application. It defines three main functions:
`echo`, `error`, and `set_verbosity`. The `echo` function writes messages to stdout if the verbosity level is enabled,
while the `error` function writes error messages to stderr. The `set_verbosity` function allows the user to toggle the
verbosity level globally.

The module uses a global variable `VERBOSE` to track the current verbosity level. This variable is used by the `echo`
function to determine whether to print messages or not. The `error` function, on the other hand, always writes error
messages to stderr, regardless of the verbosity level.

The `set_verbosity` function updates the global `VERBOSE` variable directly, which is a simple but effective way to
control the verbosity level throughout the application.
"""

from __future__ import annotations

import sys


VERBOSE = False


def echo(*args, **kwargs):
    """
    Write messages to stdout if the verbosity level is enabled.

    Parameters:
    - `*args`: The message(s) to be printed.
    - `**kwargs`: Additional keyword arguments to pass to the `print` function.

    Notes:
    The verbosity level is controlled by the global variable `VERBOSE`. If `VERBOSE` is `True`, the
    message will be printed to stdout. The `flush` argument is always set to `True` to ensure timely
    output.
    """

    if VERBOSE:
        kwargs["flush"] = True
        print(*args, **kwargs)


def error(*args, **kwargs):
    """
    Writes an error message to stderr.

    Parameters:
    - `*args`: Variable number of arguments to be joined into a single error message string.
    - `**kwargs`: Not used.

    Returns:
    - None

    Notes:
    This function always writes to stderr, regardless of the current verbosity level. The error message
    is constructed by joining the provided arguments with spaces in between. The resulting message is
    then written to stderr followed by a newline character. The underlying file descriptor is flushed
    after writing to ensure the message is immediately visible.
    """

    msg = " ".join(str(a) for a in args)
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def set_verbosity(state: bool):
    """
    Enable or disable program verbosity.

    Parameters:
    - `state`: Set verbosity state to enabled (`True`) or disabled (`False`).

    Notes:
    This updates the global `VERBOSE` variable directly, which controls the verbosity level throughout the application.
    """

    global VERBOSE

    VERBOSE = state
