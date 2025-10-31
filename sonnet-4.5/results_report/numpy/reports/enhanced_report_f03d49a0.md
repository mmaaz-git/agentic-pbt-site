# Bug Report: numpy.f2py.symbolic.replace_parenthesis Infinite Recursion and Generator Corruption on Unmatched Brackets

**Target**: `numpy.f2py.symbolic.replace_parenthesis`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `replace_parenthesis()` function enters infinite recursion when given an unmatched opening bracket, and the resulting RecursionError permanently corrupts the module-level COUNTER generator, causing all subsequent calls to fail with StopIteration.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.f2py.symbolic as symbolic

@given(st.text(alphabet='()[]{}', min_size=1, max_size=30))
def test_replace_unreplace_parenthesis_roundtrip(s):
    s_no_parens, d = symbolic.replace_parenthesis(s)
    s_restored = symbolic.unreplace_parenthesis(s_no_parens, d)
    assert s == s_restored

if __name__ == "__main__":
    test_replace_unreplace_parenthesis_roundtrip()
```

<details>

<summary>
**Failing input**: `'['`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 11, in <module>
  |     test_replace_unreplace_parenthesis_roundtrip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 5, in test_replace_unreplace_parenthesis_roundtrip
  |     def test_replace_unreplace_parenthesis_roundtrip(s):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 6, in test_replace_unreplace_parenthesis_roundtrip
    |     s_no_parens, d = symbolic.replace_parenthesis(s)
    |                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1243, in replace_parenthesis
    |     raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')
    | ValueError: Mismatch of {} parenthesis in '{{{'
    | Falsifying example: test_replace_unreplace_parenthesis_roundtrip(
    |     s='{{{',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 6, in test_replace_unreplace_parenthesis_roundtrip
    |     s_no_parens, d = symbolic.replace_parenthesis(s)
    |                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1247, in replace_parenthesis
    |     k = f'@__f2py_PARENTHESIS_{p}_{COUNTER.__next__()}@'
    |                                    ~~~~~~~~~~~~~~~~^^
    | StopIteration
    | Falsifying example: test_replace_unreplace_parenthesis_roundtrip(
    |     s='[{}',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 6, in test_replace_unreplace_parenthesis_roundtrip
    |     s_no_parens, d = symbolic.replace_parenthesis(s)
    |                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1247, in replace_parenthesis
    |     k = f'@__f2py_PARENTHESIS_{p}_{COUNTER.__next__()}@'
    |                                    ~~~~~~~~~~~~~~~~^^
    | StopIteration
    | Falsifying example: test_replace_unreplace_parenthesis_roundtrip(
    |     s='[',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

# Test case 1: Unmatched opening bracket causes RecursionError
print("Test 1: Calling replace_parenthesis('[') with unmatched opening bracket")
try:
    result = symbolic.replace_parenthesis('[')
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test case 2: After RecursionError, COUNTER is corrupted
print("\nTest 2: Calling replace_parenthesis('(a)') after the RecursionError")
try:
    result = symbolic.replace_parenthesis('(a)')
    print(f"Result: {result}")
except StopIteration as e:
    print(f"StopIteration occurred - COUNTER generator is corrupted")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")
```

<details>

<summary>
RecursionError followed by StopIteration due to COUNTER corruption
</summary>
```
Test 1: Calling replace_parenthesis('[') with unmatched opening bracket
RecursionError occurred: maximum recursion depth exceeded

Test 2: Calling replace_parenthesis('(a)') after the RecursionError
StopIteration occurred - COUNTER generator is corrupted
```
</details>

## Why This Is A Bug

This violates expected behavior because while the function's docstring states it replaces "substrings of input that are enclosed in parenthesis" (implying matched pairs), the actual implementation already includes error handling for some invalid input cases. Specifically, at line 1242-1243 in symbolic.py, the code checks for mismatched parentheses and raises a ValueError with a helpful error message. This shows the developers intended to handle invalid input gracefully rather than crash.

The bug occurs when `s.find(right, i)` returns -1 (indicating no closing bracket was found), but the code doesn't check for this before entering the while loop at line 1240. The while loop then calls `s.find(right, j + 1)` with j=-1, leading to infinite recursion at line 1249 when it calls `replace_parenthesis(s[j + len(right):])` with j=-1.

Most critically, the RecursionError exhausts the module-level COUNTER generator (defined at line 1168), permanently breaking the module for the rest of the Python session. Any subsequent call that uses COUNTER will raise StopIteration, even with valid input.

## Relevant Context

The `replace_parenthesis()` function is part of numpy's F2PY (Fortran to Python) interface generator. It processes Fortran/C expressions by temporarily replacing parenthetical subexpressions with unique placeholders during parsing. The function handles multiple parenthesis types: `()`, `[]`, `{}`, and `(/)`.

The code shows clear intent to handle errors - it already raises ValueError for inputs like `'((('` with the message "Mismatch of () parenthesis in '((('". However, it misses the case where the closing delimiter is never found (j == -1), which is what causes the infinite recursion.

The COUNTER generator is a module-level singleton used to generate unique placeholder names. Once corrupted by a RecursionError, it cannot be reset without restarting the Python interpreter, affecting all code that uses this module.

Documentation link: https://numpy.org/doc/stable/reference/generated/numpy.f2py.html

## Proposed Fix

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -1237,6 +1237,9 @@ def replace_parenthesis(s):

     i = mn_i
     j = s.find(right, i)
+
+    if j == -1:
+        raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')

     while s.count(left, i + 1, j) != s.count(right, i + 1, j):
         j = s.find(right, j + 1)
```