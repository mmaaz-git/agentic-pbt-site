# Bug Report: Cython.TestUtils._parse_pattern ValueError Crash

**Target**: `Cython.TestUtils._parse_pattern`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_pattern` function crashes with `ValueError: not enough values to unpack` when given input patterns that start with `/` or contain `:/` but lack the required closing unescaped slash delimiter.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from Cython.TestUtils import _parse_pattern

@given(st.text())
@example("/start")  # Crashes: no closing slash
@example("/")  # Crashes: empty after slash
@example(":/end")  # Crashes: end marker without closing slash
def test_parse_pattern_no_crash(pattern):
    """Property: _parse_pattern should not crash on any string input"""
    result = _parse_pattern(pattern)
    assert isinstance(result, tuple)
    assert len(result) == 3
```

**Failing inputs**: `/start`, `/`, `:/end`, `/\\/escaped` (any pattern with `/` or `:/` not followed by an unescaped `/`)

## Reproducing the Bug

```python
from Cython.TestUtils import _parse_pattern

print("Case 1: Start marker without closing slash")
try:
    result = _parse_pattern("/start")
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\nCase 2: Single slash")
try:
    result = _parse_pattern("/")
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\nCase 3: End marker without closing slash")
try:
    result = _parse_pattern(":/end")
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")
```

Output:
```
Case 1: Start marker without closing slash
ValueError: not enough values to unpack (expected 2, got 1)

Case 2: Single slash
ValueError: not enough values to unpack (expected 2, got 1)

Case 3: End marker without closing slash
ValueError: not enough values to unpack (expected 2, got 1)
```

## Why This Is A Bug

The function is used to parse test directive patterns in Cython's test framework (see line 244 and 251 in TestUtils.py where it's called). If a test contains a malformed pattern directive, the test framework will crash with an unhelpful ValueError instead of gracefully handling or reporting the malformed pattern.

**Root cause**: Lines 196 and 201 in TestUtils.py use `re.split()` with `maxsplit=1`, which returns a list with only ONE element when no delimiter is found, but the code tries to unpack to TWO variables:

```python
start, pattern = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
```

When `re.split` finds no unescaped `/` in the string, it returns `['the_whole_string']`, causing the unpacking to fail.

## Fix

```diff
--- a/Cython/TestUtils.py
+++ b/Cython/TestUtils.py
@@ -193,11 +193,15 @@ _strip_cython_code_from_html = partial(re.compile(
 def _parse_pattern(pattern):
     start = end = None
     if pattern.startswith('/'):
-        start, pattern = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
+        parts = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
+        if len(parts) == 2:
+            start, pattern = parts
+        else:
+            start, pattern = parts[0], ''
         pattern = pattern.strip()
     if pattern.startswith(':'):
         pattern = pattern[1:].strip()
         if pattern.startswith("/"):
-            end, pattern = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
+            parts = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
+            if len(parts) == 2:
+                end, pattern = parts
+            else:
+                end, pattern = parts[0], ''
             pattern = pattern.strip()
     return start, end, pattern
```