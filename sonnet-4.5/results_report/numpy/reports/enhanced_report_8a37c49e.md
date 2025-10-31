# Bug Report: numpy.char Functions Incorrectly Handle Null Character Searches

**Target**: `numpy.char.find`, `numpy.char.rfind`, `numpy.char.startswith`, `numpy.char.endswith`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Multiple numpy.char string search functions systematically return incorrect results when searching for null characters (`'\x00'`), violating their documented behavior of matching Python's str methods element-wise.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=0, max_size=50), min_size=1, max_size=10))
@settings(max_examples=500)
def test_find_matches_python_str(strings):
    arr = np.array(strings)
    numpy_results = char.find(arr, '\x00')

    for i, s in enumerate(strings):
        python_result = s.find('\x00')
        assert numpy_results[i] == python_result, f"find mismatch for {s!r}: numpy={numpy_results[i]}, python={python_result}"

# Run the test
if __name__ == "__main__":
    test_find_matches_python_str()
```

<details>

<summary>
**Failing input**: `strings=['']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 17, in <module>
    test_find_matches_python_str()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 6, in test_find_matches_python_str
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 13, in test_find_matches_python_str
    assert numpy_results[i] == python_result, f"find mismatch for {s!r}: numpy={numpy_results[i]}, python={python_result}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: find mismatch for '': numpy=0, python=-1
Falsifying example: test_find_matches_python_str(
    strings=[''],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

# Test with empty string
arr = np.array([''])

print("Testing numpy.char functions with null character on empty string")
print("=" * 60)

print("\nfind():")
print(f"  numpy.char.find([''], '\\x00') = {char.find(arr, '\x00')[0]}")
print(f"  Python ''.find('\\x00')        = {''.find(chr(0))}")

print("\nrfind():")
print(f"  numpy.char.rfind([''], '\\x00') = {char.rfind(arr, '\x00')[0]}")
print(f"  Python ''.rfind('\\x00')        = {''.rfind(chr(0))}")

print("\nstartswith():")
print(f"  numpy.char.startswith([''], '\\x00') = {char.startswith(arr, '\x00')[0]}")
print(f"  Python ''.startswith('\\x00')        = {''.startswith(chr(0))}")

print("\nendswith():")
print(f"  numpy.char.endswith([''], '\\x00') = {char.endswith(arr, '\x00')[0]}")
print(f"  Python ''.endswith('\\x00')        = {''.endswith(chr(0))}")

# Test with more cases
print("\n" + "=" * 60)
print("Testing with various string patterns:")
print("=" * 60)

test_cases = [
    '',           # empty string
    'a',          # string without null
    '\x00',       # just null character
    'a\x00b',     # null in middle
    '\x00ab',     # null at start
    'ab\x00'      # null at end
]

for test_str in test_cases:
    arr = np.array([test_str])
    print(f"\nString: {test_str!r}")
    print(f"  find:       numpy={char.find(arr, '\x00')[0]:3d}, python={test_str.find('\x00'):3d}")
    print(f"  rfind:      numpy={char.rfind(arr, '\x00')[0]:3d}, python={test_str.rfind('\x00'):3d}")
    print(f"  startswith: numpy={char.startswith(arr, '\x00')[0]!s:5s}, python={test_str.startswith('\x00')!s:5s}")
    print(f"  endswith:   numpy={char.endswith(arr, '\x00')[0]!s:5s}, python={test_str.endswith('\x00')!s:5s}")
```

<details>

<summary>
Output showing systematic incorrect behavior across all test cases
</summary>
```
Testing numpy.char functions with null character on empty string
============================================================

find():
  numpy.char.find([''], '\x00') = 0
  Python ''.find('\x00')        = -1

rfind():
  numpy.char.rfind([''], '\x00') = 0
  Python ''.rfind('\x00')        = -1

startswith():
  numpy.char.startswith([''], '\x00') = True
  Python ''.startswith('\x00')        = False

endswith():
  numpy.char.endswith([''], '\x00') = True
  Python ''.endswith('\x00')        = False

============================================================
Testing with various string patterns:
============================================================

String: ''
  find:       numpy=  0, python= -1
  rfind:      numpy=  0, python= -1
  startswith: numpy=True , python=False
  endswith:   numpy=True , python=False

String: 'a'
  find:       numpy=  0, python= -1
  rfind:      numpy=  1, python= -1
  startswith: numpy=True , python=False
  endswith:   numpy=True , python=False

String: '\x00'
  find:       numpy=  0, python=  0
  rfind:      numpy=  0, python=  0
  startswith: numpy=True , python=True
  endswith:   numpy=True , python=True

String: 'a\x00b'
  find:       numpy=  0, python=  1
  rfind:      numpy=  3, python=  1
  startswith: numpy=True , python=False
  endswith:   numpy=True , python=False

String: '\x00ab'
  find:       numpy=  0, python=  0
  rfind:      numpy=  3, python=  0
  startswith: numpy=True , python=True
  endswith:   numpy=True , python=False

String: 'ab\x00'
  find:       numpy=  0, python=  2
  rfind:      numpy=  2, python=  2
  startswith: numpy=True , python=False
  endswith:   numpy=True , python=True
```
</details>

## Why This Is A Bug

This violates the explicit API contract documented in multiple places:

1. **Documentation contract violation**: Each function's docstring explicitly references its Python str counterpart in the "See Also" section:
   - `numpy.char.find`: "See Also: str.find"
   - `numpy.char.rfind`: "See Also: str.rfind"
   - `numpy.char.startswith`: "See Also: str.startswith"
   - `numpy.char.endswith`: "See Also: str.endswith"

2. **Systematic incorrect behavior**: The functions exhibit consistent patterns of errors:
   - `find()` often returns 0 (found at position 0) when it should return -1 (not found)
   - `rfind()` returns incorrect positions, sometimes the string length instead of -1
   - `startswith()` and `endswith()` almost always return True, even when False is correct

3. **Not limited to edge cases**: The bug affects common cases like empty strings and strings without null characters, not just obscure inputs

4. **Data correctness impact**: Any code relying on these functions for string search operations involving null bytes will produce incorrect results, potentially leading to logic errors, incorrect data processing, or security issues in applications that depend on accurate string matching.

## Relevant Context

The numpy.char module is implemented as a wrapper around numpy.strings (previously numpy._core.strings), which in turn uses C-implemented ufuncs (`_find_ufunc`, `_rfind_ufunc`, `_startswith_ufunc`, `_endswith_ufunc` from numpy._core.umath).

The issue appears to stem from C-style null-terminated string handling, where null bytes (`\x00`) are treated as string terminators rather than regular characters. This causes:
- Empty strings to be treated as if they contain/start with/end with null
- Positions after null bytes to be incorrectly reported
- String lengths to be miscalculated when null bytes are present

**Important Note**: numpy.char is marked as a legacy module in the NumPy documentation with a warning that it will no longer receive updates. Users are advised to use numpy.strings instead. However, the module is still included in NumPy and actively used by existing codebases.

Relevant documentation:
- numpy.char module: https://numpy.org/doc/stable/reference/routines.char.html
- Python str.find: https://docs.python.org/3/library/stdtypes.html#str.find
- numpy.strings (recommended alternative): https://numpy.org/doc/stable/reference/routines.strings.html

## Proposed Fix

Since the bug is in C-level ufunc implementations that treat null as a string terminator, fixing requires modifying the underlying C code. However, given numpy.char's legacy status, a documentation fix may be more appropriate:

```diff
--- a/numpy/_core/defchararray.py
+++ b/numpy/_core/defchararray.py
@@ -10,6 +10,11 @@
    in the `numpy.char` module for fast vectorized string operations.

+.. warning::
+   The numpy.char module has known limitations with null character (`'\x00'`)
+   handling due to C-style string processing. Functions may produce incorrect
+   results when searching for or processing null bytes. Use numpy.strings for
+   correct null character handling.
+
 Some methods will only be available if the corresponding string method is
 available in your version of Python.
```

For users needing correct behavior immediately, use numpy.strings directly or Python's native string methods with list comprehensions:
```python
# Instead of: numpy.char.find(arr, '\x00')
# Use: [s.find('\x00') for s in arr]
# Or: numpy.strings.find(arr, '\x00')  # If available in your NumPy version
```