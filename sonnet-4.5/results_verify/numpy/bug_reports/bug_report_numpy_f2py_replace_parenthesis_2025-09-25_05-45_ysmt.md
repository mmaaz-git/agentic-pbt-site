# Bug Report: numpy.f2py.symbolic.replace_parenthesis Infinite Recursion on Unmatched Opening Parenthesis

**Target**: `numpy.f2py.symbolic.replace_parenthesis`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `replace_parenthesis` function enters infinite recursion when given a string with an unmatched opening parenthesis, leading to a RecursionError instead of raising a proper ValueError.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.f2py.symbolic as symbolic

@given(st.text())
def test_replace_unreplace_parenthesis_roundtrip(s):
    new_s, mapping = symbolic.replace_parenthesis(s)
    restored = symbolic.unreplace_parenthesis(new_s, mapping)
    assert restored == s
```

**Failing input**: `s='('`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

symbolic.replace_parenthesis('(')
```

This causes infinite recursion and eventually:
```
RecursionError: maximum recursion depth exceeded
```

## Why This Is A Bug

The function correctly raises ValueError for other mismatched parenthesis cases (e.g., `'((('`), but fails to detect the case where there's an unmatched opening parenthesis with no corresponding closing parenthesis.

The bug occurs because:
1. The function finds the opening `'('` at index 0
2. It looks for the closing `')'` but `j = s.find(')', 0)` returns -1
3. The while loop condition checks `s.count('(', 1, -1) != s.count(')', 1, -1)` which is `0 != 0`, so it doesn't enter the loop
4. It then recursively calls `replace_parenthesis(s[j + len(right):])` which becomes `replace_parenthesis(s[0:])` = `replace_parenthesis('(')`
5. This creates infinite recursion

## Fix

Add a check after the while loop to detect when no closing delimiter was found:

```diff
--- a/symbolic.py
+++ b/symbolic.py
@@ -1240,6 +1240,8 @@ def replace_parenthesis(s):
     while s.count(left, i + 1, j) != s.count(right, i + 1, j):
         j = s.find(right, j + 1)
         if j == -1:
             raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')
+    if j == -1:
+        raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')

     p = {'(': 'ROUND', '[': 'SQUARE', '{': 'CURLY', '(/': 'ROUNDDIV'}[left]
```