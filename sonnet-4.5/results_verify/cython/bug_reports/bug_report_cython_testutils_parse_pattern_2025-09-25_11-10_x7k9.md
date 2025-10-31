# Bug Report: Cython.TestUtils._parse_pattern ValueError on Missing Delimiter

**Target**: `Cython.TestUtils._parse_pattern`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_parse_pattern` function crashes with a `ValueError` when given patterns that start with `/` or `:/` but lack the required second delimiter. The function uses `re.split()` with tuple unpacking, which fails when the delimiter is not found.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from Cython.TestUtils import _parse_pattern

@given(st.text(min_size=1, max_size=50))
@example("/start")
@example("/")
@example(":/end")
@example(":/")
def test_parse_pattern_should_not_crash(pattern):
    start, end, parsed = _parse_pattern(pattern)
    assert isinstance(start, (str, type(None)))
    assert isinstance(end, (str, type(None)))
    assert isinstance(parsed, str)
```

**Failing inputs**:
- `/start` (no second `/`)
- `/` (empty start marker)
- `:/end` (no second `/` after `:`)
- `:/` (empty end marker)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.TestUtils import _parse_pattern

result = _parse_pattern("/start")
```

This raises:
```
ValueError: not enough values to unpack (expected 2, got 1)
```

## Why This Is A Bug

The function is supposed to parse pattern strings for test assertions in Cython's testing framework. The format supports optional start and end markers (e.g., `/start/:/end/pattern`). However, when a user provides a malformed pattern like `/start` (missing the closing `/`), the function crashes instead of either:
1. Gracefully handling the error
2. Treating the entire string as the pattern

The existing unit tests only cover well-formed patterns and miss these edge cases.

## Fix

```diff
--- a/TestUtils.py
+++ b/TestUtils.py
@@ -193,13 +193,18 @@ def _parse_pattern(pattern):
 def _parse_pattern(pattern):
     start = end = None
     if pattern.startswith('/'):
-        start, pattern = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
+        parts = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
+        if len(parts) == 2:
+            start, pattern = parts
+        else:
+            pattern = parts[0]
         pattern = pattern.strip()
     if pattern.startswith(':'):
         pattern = pattern[1:].strip()
         if pattern.startswith("/"):
-            end, pattern = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
+            parts = re.split(r"(?<!\\)/", pattern[1:], maxsplit=1)
+            if len(parts) == 2:
+                end, pattern = parts
             pattern = pattern.strip()
     return start, end, pattern
```