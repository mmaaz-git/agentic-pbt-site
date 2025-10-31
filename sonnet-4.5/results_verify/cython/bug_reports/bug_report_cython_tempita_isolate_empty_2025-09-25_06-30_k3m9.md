# Bug Report: Cython.Tempita isolate_expression Empty String IndexError

**Target**: `Cython.Tempita._tempita.isolate_expression`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `isolate_expression` function crashes with IndexError when called on an empty string, as it doesn't handle the case where `splitlines()` returns an empty list.

## Property-Based Test

```python
@given(st.integers(min_value=1, max_value=10),
       st.integers(min_value=0, max_value=10))
@settings(max_examples=100)
def test_isolate_expression_handles_empty_string(row, col):
    text = ""

    try:
        result = isolate_expression(text, (row, col), (row, col))
    except IndexError:
        assert False, "isolate_expression should handle empty strings gracefully"
```

**Failing input**: Empty string with any position, e.g., `isolate_expression("", (1, 0), (1, 0))`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._tempita import isolate_expression

text = ""

try:
    result = isolate_expression(text, (1, 0), (1, 0))
    print(f"Result: {result!r}")
except IndexError as e:
    print(f"IndexError: {e}")
    print(f"Bug: Empty string causes list index out of range")
```

## Why This Is A Bug

Line 1019 in `Cython/Tempita/_tempita.py` calls `lines = string.splitlines(True)`, which returns an empty list `[]` when string is empty.

Lines 1020-1021 then attempt to access `lines[srow]` without checking if the list is empty:
```python
if srow == erow:
    return lines[srow][scol:ecol]
```

Since srow is computed as `start_pos[0] - 1` (line 1016), if start_pos is `(1, 0)`, then srow is 0. Accessing `lines[0]` on an empty list raises IndexError.

While empty strings are rare in practice, the function is called internally by `parse_signature` during template parsing, so malformed templates could trigger this code path.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -1017,6 +1017,8 @@ def isolate_expression(string, start_pos, end_pos):
     erow, ecol = end_pos
     erow -= 1
     lines = string.splitlines(True)
+    if not lines:
+        return ''
     if srow == erow:
         return lines[srow][scol:ecol]
     parts = [lines[srow][scol:]]
```