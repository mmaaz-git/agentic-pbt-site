# Bug Report: pandas.io.clipboard read_clipboard() Last Line Loss

**Target**: `pandas.io.clipboards.read_clipboard()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_clipboard()` function incorrectly discards the last line of clipboard data when the text does not end with a newline character. This occurs during the tab detection logic that determines whether the clipboard contains tab-separated data.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example, settings


@st.composite
def lines_strategy(draw, min_lines=2, max_lines=10):
    num_lines = draw(st.integers(min_value=min_lines, max_value=max_lines))
    lines = [
        draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters='\n\r'), min_size=1, max_size=50))
        for _ in range(num_lines)
    ]
    return lines


@example(["a\tb", "c\td", "e\tf"])
@example(["line1", "line2", "line3", "line4"])
@given(lines_strategy(min_lines=2, max_lines=10))
@settings(max_examples=200)
def test_line_count_preserved_without_trailing_newline(lines):
    text_without_newline = "\n".join(lines)
    text_with_newline = text_without_newline + "\n"

    processed_without = text_without_newline[:10000].split("\n")[:-1][:10]
    processed_with = text_with_newline[:10000].split("\n")[:-1][:10]

    expected_lines = lines[:10]

    assert processed_with == expected_lines, (
        f"With trailing newline: expected {expected_lines}, got {processed_with}"
    )

    assert processed_without == expected_lines, (
        f"WITHOUT trailing newline: expected {expected_lines}, got {processed_without}. "
        f"Last line '{lines[-1]}' was lost! This is a bug."
    )
```

**Failing input**: `["a\tb", "c\td", "e\tf"]` (and many others)

## Reproducing the Bug

```python
text_with_newline = "a\tb\nc\td\ne\tf\n"
text_without_newline = "a\tb\nc\td\ne\tf"

processed_with = text_with_newline[:10000].split("\n")[:-1][:10]
processed_without = text_without_newline[:10000].split("\n")[:-1][:10]

print(f"With newline: {processed_with}")
print(f"Without newline: {processed_without}")

assert processed_with == ["a\tb", "c\td", "e\tf"]
assert processed_without == ["a\tb", "c\td"]
```

## Why This Is A Bug

The tab detection logic at line 98 in `clipboards.py` assumes that clipboard text always ends with a newline:

```python
lines = text[:10000].split("\n")[:-1][:10]
```

When text ends with `\n`:
- `"a\nb\nc\n".split("\n")` → `["a", "b", "c", ""]`
- `[:-1]` removes the empty string → `["a", "b", "c"]` ✓

When text does NOT end with `\n`:
- `"a\nb\nc".split("\n")` → `["a", "b", "c"]`
- `[:-1]` removes the last line → `["a", "b"]` ✗

This causes data loss when clipboard text doesn't have a trailing newline, which is common when copying text from applications that don't add trailing newlines.

## Fix

```diff
--- a/pandas/io/clipboards.py
+++ b/pandas/io/clipboards.py
@@ -95,7 +95,11 @@ def read_clipboard(
     # inspect no more then the 10 first lines, if they
     # all contain an equal number (>0) of tabs, infer
     # that this came from excel and set 'sep' accordingly
-    lines = text[:10000].split("\n")[:-1][:10]
+    split_lines = text[:10000].split("\n")
+    if text[:10000].endswith("\n"):
+        lines = split_lines[:-1][:10]
+    else:
+        lines = split_lines[:10]

     # Need to remove leading white space, since read_csv
     # accepts:
```

Alternatively, a more concise fix:

```diff
--- a/pandas/io/clipboards.py
+++ b/pandas/io/clipboards.py
@@ -95,7 +95,7 @@ def read_clipboard(
     # inspect no more then the 10 first lines, if they
     # all contain an equal number (>0) of tabs, infer
     # that this came from excel and set 'sep' accordingly
-    lines = text[:10000].split("\n")[:-1][:10]
+    lines = [line for line in text[:10000].split("\n")[:10] if line or text[:10000].endswith("\n")][:10]
```

Or the simplest fix:

```diff
--- a/pandas/io/clipboards.py
+++ b/pandas/io/clipboards.py
@@ -95,7 +95,7 @@ def read_clipboard(
     # inspect no more then the 10 first lines, if they
     # all contain an equal number (>0) of tabs, infer
     # that this came from excel and set 'sep' accordingly
-    lines = text[:10000].split("\n")[:-1][:10]
+    lines = text[:10000].rstrip("\n").split("\n")[:10]
```