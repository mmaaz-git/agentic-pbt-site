# Bug Report: numpy.f2py.symbolic.replace_parenthesis Infinite Recursion on Unmatched Parenthesis

**Target**: `numpy.f2py.symbolic.replace_parenthesis`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`replace_parenthesis()` enters infinite recursion when given a string with unmatched opening parenthesis, instead of raising the intended ValueError.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.f2py.symbolic as symbolic

@given(st.text())
def test_parenthesis_round_trip(s):
    new_s, d = symbolic.replace_parenthesis(s)
    reconstructed = symbolic.unreplace_parenthesis(new_s, d)
    assert reconstructed == s
```

**Failing input**: `s="("`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

s = "("
new_s, d = symbolic.replace_parenthesis(s)
```

## Why This Is A Bug

When an opening parenthesis has no matching closing parenthesis:
1. `j = s.find(')', i)` returns -1
2. The while loop checking balanced parenthesis: `while s.count('(', i+1, j) != s.count(')', i+1, j)`
   - With j=-1, slicing uses -1 as end index
   - Both counts are 0, so the loop exits immediately
3. The ValueError check is inside the while loop: `if j == -1: raise ValueError(...)`
4. Since the loop never executes, the ValueError is never raised
5. The function proceeds to recursively call itself with the same input: `replace_parenthesis(s[j + len(right):])`
6. This becomes `replace_parenthesis(s[0:])` which is `replace_parenthesis("(")` again
7. Infinite recursion until RecursionError

## Fix

Move the ValueError check outside the while loop:

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -1237,10 +1237,11 @@ def replace_parenthesis(s):
     i = mn_i
     j = s.find(right, i)

     while s.count(left, i + 1, j) != s.count(right, i + 1, j):
         j = s.find(right, j + 1)
-        if j == -1:
-            raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')
+
+    if j == -1:
+        raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')

     p = {'(': 'ROUND', '[': 'SQUARE', '{': 'CURLY', '(/': 'ROUNDDIV'}[left]
```