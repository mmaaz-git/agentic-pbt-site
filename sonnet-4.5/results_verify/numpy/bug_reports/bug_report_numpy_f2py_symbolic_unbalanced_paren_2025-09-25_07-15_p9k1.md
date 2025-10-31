# Bug Report: numpy.f2py.symbolic Crash on Unbalanced Parentheses

**Target**: `numpy.f2py.symbolic.fromstring`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Parsing strings with unbalanced opening parentheses causes infinite recursion and crashes instead of raising a proper `ValueError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy.f2py.symbolic as sym

@given(st.text(min_size=1, max_size=50, alphabet='abcdefghijklmnopqrstuvwxyz +*-()0123456789'))
@settings(max_examples=500)
def test_fromstring_does_not_crash(s):
    try:
        expr = sym.fromstring(s)
    except (ValueError, KeyError, RecursionError, AttributeError):
        assume(False)
```

**Failing input**: `'('`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as sym

try:
    expr = sym.fromstring("(")
    print(f"Parsed: {expr}")
except RecursionError as e:
    print(f"RecursionError raised (BUG)")
except ValueError as e:
    print(f"ValueError raised (expected): {e}")
```

Output:
```
RecursionError raised (BUG)
```

Other unbalanced inputs:
- `"("` → RecursionError (bug)
- `"(("` → RecursionError (bug)
- `"(()"` → ValueError (correct behavior)
- `")"` → Parses as symbol (questionable but not crashing)

## Why This Is A Bug

Malformed input should raise `ValueError` with a clear error message, as it does for some unbalanced cases like `"(()"`. Instead, single or double opening parentheses cause infinite recursion and crash.

The function `replace_parenthesis` in `symbolic.py` enters an infinite loop when it cannot find a matching closing parenthesis. The `while` loop at line ~1234 uses `j` (which is -1 when no match is found) in `s.count(left, i + 1, j)`, causing incorrect behavior.

## Fix

Check for unmatched opening parentheses before entering the while loop:

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -1232,6 +1232,9 @@ def replace_parenthesis(s):
     i = mn_i
     j = s.find(right, i)

+    if j == -1:
+        raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')
+
     while s.count(left, i + 1, j) != s.count(right, i + 1, j):
         j = s.find(right, j + 1)
         if j == -1:
```