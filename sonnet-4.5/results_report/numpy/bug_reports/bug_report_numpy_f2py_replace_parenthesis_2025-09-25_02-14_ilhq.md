# Bug Report: numpy.f2py.symbolic.replace_parenthesis RecursionError on Unbalanced Delimiters

**Target**: `numpy.f2py.symbolic.replace_parenthesis`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `replace_parenthesis` function enters infinite recursion when given input with unbalanced opening delimiters (e.g., `'('`, `'['`, `'{'`), leading to RecursionError.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from numpy.f2py import symbolic


@given(st.text(min_size=1, max_size=200))
@settings(max_examples=500)
def test_parenthesis_replacement_round_trip(s):
    new_s, mapping = symbolic.replace_parenthesis(s)
    reconstructed = symbolic.unreplace_parenthesis(new_s, mapping)
    assert s == reconstructed
```

**Failing input**: `s='('` (single opening parenthesis)

## Reproducing the Bug

```python
from numpy.f2py import symbolic

s = '('
result = symbolic.replace_parenthesis(s)
```

Output:
```
RecursionError: maximum recursion depth exceeded
```

## Why This Is A Bug

When the closing delimiter is not found (`j = s.find(right, i)` returns -1), the code has a check inside the while loop to raise a ValueError. However, this check is only executed if the while loop runs at least once.

When `j = -1`, the while loop condition `s.count(left, i+1, -1) != s.count(right, i+1, -1)` evaluates to `0 != 0` (False), so the loop body never executes and the ValueError is never raised.

Instead, the code proceeds to the recursive call: `replace_parenthesis(s[j + len(right):])` which becomes `replace_parenthesis(s[0:])` = `replace_parenthesis(s)`, recursively calling itself with the same input infinitely.

This affects any input with unbalanced opening delimiters: `(`, `[`, `{`, or `(/`.

## Fix

Move the mismatch check outside the while loop to ensure it executes when no closing delimiter is found:

```diff
--- a/symbolic.py
+++ b/symbolic.py
@@ -1235,11 +1235,11 @@ def replace_parenthesis(s):

     i = mn_i
     j = s.find(right, i)
+    if j == -1:
+        raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')

     while s.count(left, i + 1, j) != s.count(right, i + 1, j):
         j = s.find(right, j + 1)
-        if j == -1:
-            raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')

     p = {'(': 'ROUND', '[': 'SQUARE', '{': 'CURLY', '(/': 'ROUNDDIV'}[left]
```