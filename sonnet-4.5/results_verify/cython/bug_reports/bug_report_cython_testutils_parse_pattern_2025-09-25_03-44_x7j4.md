# Bug Report: Cython.TestUtils _parse_pattern ValueError with Backslash

**Target**: `Cython.TestUtils._parse_pattern`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_pattern` function crashes with a `ValueError: not enough values to unpack` when the pattern starts with a slash followed by a backslash and no subsequent unescaped slash (e.g., `/\/`).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.TestUtils import _parse_pattern

@given(
    st.text(alphabet=st.characters(blacklist_characters='/'), min_size=0),
    st.text(min_size=1)
)
def test_parse_pattern_with_start_marker(start_marker, pattern):
    full_pattern = f"/{start_marker}/{pattern}"
    start, end, parsed = _parse_pattern(full_pattern)
    assert start == start_marker
```

**Failing input**: `start_marker='\\'`, `pattern='0'` (or any pattern)

## Reproducing the Bug

```python
from Cython.TestUtils import _parse_pattern

_parse_pattern("/\\/")
```

Output:
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".../Cython/TestUtils.py", line 196, in _parse_pattern
    start, pattern = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
ValueError: not enough values to unpack (expected 2, got 1)
```

## Why This Is A Bug

The function is designed to parse pattern strings with optional start/end delimiters using the format `/start/pattern` or `/start/:/end/pattern`. However, when the input is `/\/` (a start marker consisting of a backslash), the regex `r"(?<!\\)/"` (which matches `/` not preceded by `\\`) doesn't find a match in the substring `\/`, causing `re.split()` to return a list with only one element instead of two.

This crashes the unpacking on line 196:
```python
start, pattern = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
```

## Fix

```diff
--- a/Cython/TestUtils.py
+++ b/Cython/TestUtils.py
@@ -193,7 +193,10 @@ def _parse_pattern(pattern):
 def _parse_pattern(pattern):
     start = end = None
     if pattern.startswith('/'):
-        start, pattern = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
+        parts = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
+        if len(parts) < 2:
+            return None, None, pattern
+        start, pattern = parts
         pattern = pattern.strip()
     if pattern.startswith(':'):
         pattern = pattern[1:].strip()