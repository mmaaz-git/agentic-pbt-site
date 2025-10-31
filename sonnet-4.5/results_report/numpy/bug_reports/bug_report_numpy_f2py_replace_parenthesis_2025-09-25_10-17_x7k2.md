# Bug Report: numpy.f2py.symbolic.replace_parenthesis RecursionError and COUNTER Corruption

**Target**: `numpy.f2py.symbolic.replace_parenthesis`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`replace_parenthesis()` causes infinite recursion on unmatched opening brackets, and the resulting RecursionError corrupts the module-level COUNTER generator, breaking all subsequent calls to functions that use it.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.f2py.symbolic as symbolic

@given(st.text(alphabet='()[]{}', min_size=1, max_size=30))
def test_replace_unreplace_parenthesis_roundtrip(s):
    s_no_parens, d = symbolic.replace_parenthesis(s)
    s_restored = symbolic.unreplace_parenthesis(s_no_parens, d)
    assert s == s_restored
```

**Failing input**: `'['`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

symbolic.replace_parenthesis('[')
```

This raises RecursionError. Subsequently:

```python
symbolic.replace_parenthesis('(a)')
```

This now raises StopIteration because COUNTER is corrupted.

## Why This Is A Bug

When `replace_parenthesis()` encounters an unmatched opening bracket like `'['`, it searches for the closing bracket with `j = s.find(']', i)`, which returns -1. The function then enters an infinite recursion because:

1. The while loop condition uses `s.count('[', i+1, j)` with j=-1
2. This causes incorrect recursion in `replace_parenthesis(s[j + len(right):])`
3. The RecursionError corrupts the module-level COUNTER generator
4. All subsequent calls using COUNTER raise StopIteration

This is a severe bug because one failed call permanently breaks the module.

## Fix

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -1234,6 +1234,9 @@ def replace_parenthesis(s):

     i = mn_i
     j = s.find(right, i)
+
+    if j == -1:
+        raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')

     while s.count(left, i + 1, j) != s.count(right, i + 1, j):
         j = s.find(right, j + 1)
```