#!/usr/bin/env python3
"""
The JS comment extractor must tolerate how the model wraps its reply:
a /** */ block (preferred), a ``` fenced block, or plain text. It must also
preserve genuine content bullets ('* item') rather than eating them.

Guards two issues: content asterisks being stripped, and class constructors
falling back to 'documentation generation failed' when the model used a
```js fence instead of /** */.

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_javascript import _extract_first_comment_block as ex  # noqa: E402


def main():
    # 1. JSDoc block + content bullet preserved.
    d = ex("/**\n * Does a thing. Options:\n * * fast mode\n */")
    assert "Does a thing" in d and "* fast mode" in d, d

    # 2. A ```javascript fence (the exact shape that broke the Counter constructor).
    fenced = (
        "```javascript\n"
        "Initialize a new instance of the Counter class with a specified starting value.\n"
        "\n"
        "Parameters:\n"
        "- start: The initial value for the counter.\n"
        "```"
    )
    d2 = ex(fenced)
    assert "Initialize a new instance" in d2, d2
    assert "```" not in d2, f"fence leaked into output: {d2!r}"

    # 3. Plain text, no fences at all.
    assert ex("Simply returns the sum of its two arguments.") == "Simply returns the sum of its two arguments."

    # 4. Genuinely empty reply -> empty string.
    assert ex("   \n  \n") == ""

    print("PASS: extractor accepts JSDoc, fenced, and plain-text replies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
