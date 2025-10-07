#!/usr/bin/env python3
"""
This module TODO.
"""

from __future__ import annotations

import sys


VERBOSE = False


def echo(*args, **kwargs):
    """Write something to stdout if `VERBOSE is True`."""
    if VERBOSE:
        kwargs["flush"] = True
        print(*args, **kwargs)


def error(*args, **kwargs):
    """Write an error message to stderr."""
    msg = " ".join(str(a) for a in args)
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def set_verbosity(state: bool):
    global VERBOSE

    VERBOSE = state
