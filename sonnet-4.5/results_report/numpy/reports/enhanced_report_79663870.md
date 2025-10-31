# Bug Report: numpy.f2py.symbolic.replace_parenthesis Infinite Recursion on Unmatched Opening Parenthesis

**Target**: `numpy.f2py.symbolic.replace_parenthesis`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `replace_parenthesis` function enters infinite recursion when given a string containing a single unmatched opening parenthesis, causing a RecursionError instead of raising the expected ValueError.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
import numpy.f2py.symbolic as symbolic

@given(st.text())
@example('(')
@settings(max_examples=10)
def test_replace_unreplace_parenthesis_roundtrip(s):
    new_s, mapping = symbolic.replace_parenthesis(s)
    restored = symbolic.unreplace_parenthesis(new_s, mapping)
    assert restored == s

if __name__ == "__main__":
    test_replace_unreplace_parenthesis_roundtrip()
```

<details>

<summary>
**Failing input**: `s='('`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 13, in <module>
    test_replace_unreplace_parenthesis_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 5, in test_replace_unreplace_parenthesis_roundtrip
    @example('(')

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 8, in test_replace_unreplace_parenthesis_roundtrip
    new_s, mapping = symbolic.replace_parenthesis(s)
                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1249, in replace_parenthesis
    r, d = replace_parenthesis(s[j + len(right):])
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1249, in replace_parenthesis
    r, d = replace_parenthesis(s[j + len(right):])
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1249, in replace_parenthesis
    r, d = replace_parenthesis(s[j + len(right):])
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  [Previous line repeated 1996 more times]
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1247, in replace_parenthesis
    k = f'@__f2py_PARENTHESIS_{p}_{COUNTER.__next__()}@'
                                   ~~~~~~~~~~~~~~~~^^
RecursionError: maximum recursion depth exceeded
Falsifying explicit example: test_replace_unreplace_parenthesis_roundtrip(
    s='(',
)
```
</details>

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

result = symbolic.replace_parenthesis('(')
print(result)
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/repo.py", line 3, in <module>
    result = symbolic.replace_parenthesis('(')
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1249, in replace_parenthesis
    r, d = replace_parenthesis(s[j + len(right):])
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1249, in replace_parenthesis
    r, d = replace_parenthesis(s[j + len(right):])
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1249, in replace_parenthesis
    r, d = replace_parenthesis(s[j + len(right):])
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  [Previous line repeated 995 more times]
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1247, in replace_parenthesis
    k = f'@__f2py_PARENTHESIS_{p}_{COUNTER.__next__()}@'
                                   ~~~~~~~~~~~~~~~~^^
RecursionError: maximum recursion depth exceeded
```
</details>

## Why This Is A Bug

The function is designed to detect and report mismatched parentheses by raising a ValueError, as evidenced by the error handling code at lines 1242-1243. The function successfully raises ValueError for other mismatched cases like `'((('`, but fails for a single unmatched opening parenthesis `'('`.

The infinite recursion occurs because:
1. The function finds the opening `'('` at index 0
2. It searches for the closing `')'` with `j = s.find(')', i)`, which returns -1 (not found)
3. The while loop condition `s.count('(', i + 1, j) != s.count(')', i + 1, j)` evaluates to `s.count('(', 1, -1) != s.count(')', 1, -1)`, which is `0 != 0` (False), so the loop doesn't execute
4. The error check inside the while loop (lines 1242-1243) is never reached
5. The function then calls `replace_parenthesis(s[j + len(right):])`, which with j=-1 and right=')' becomes `replace_parenthesis(s[0:])`, i.e., `replace_parenthesis('(')`
6. This creates infinite recursion

This violates the expected behavior of consistent error handling for all mismatched parenthesis cases.

## Relevant Context

The `replace_parenthesis` function is part of numpy's f2py module (Fortran to Python Interface Generator), used internally for parsing Fortran code. While a single unmatched parenthesis is unlikely in valid Fortran code, the function should handle all error cases consistently to prevent crashes.

The function correctly handles:
- Multiple unmatched opening parentheses: `'((('` → raises ValueError
- Unmatched closing parentheses: `')))'` → returns normally (no parentheses to replace)
- Properly matched parentheses: `'()'` → works correctly

But fails on:
- Single unmatched opening parenthesis: `'('` → infinite recursion

## Proposed Fix

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