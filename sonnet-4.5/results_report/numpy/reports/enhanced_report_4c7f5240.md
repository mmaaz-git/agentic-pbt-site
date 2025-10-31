# Bug Report: numpy.strings.replace Silently Truncates When Replacing Entire String With Longer Replacement

**Target**: `numpy.strings.replace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.replace()` silently returns the original string unchanged when attempting to replace the entire string with a longer replacement, violating its documented promise to match Python's `str.replace()` behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import numpy.strings as nps

@given(st.lists(st.text(min_size=1), min_size=1, max_size=10), st.text(min_size=1, max_size=5), st.text(max_size=5))
def test_replace_matches_python(strings, old, new):
    for s in strings:
        if old in s:
            arr = np.array([s])
            np_result = nps.replace(arr, old, new)[0]
            py_result = s.replace(old, new)
            assert np_result == py_result, f"Failed for s='{s}', old='{old}', new='{new}': expected '{py_result}' but got '{np_result}'"

if __name__ == "__main__":
    test_replace_matches_python()
```

<details>

<summary>
**Failing input**: `strings=['0'], old='0', new='00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 15, in <module>
    test_replace_matches_python()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 6, in test_replace_matches_python
    def test_replace_matches_python(strings, old, new):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 12, in test_replace_matches_python
    assert np_result == py_result, f"Failed for s='{s}', old='{old}', new='{new}': expected '{py_result}' but got '{np_result}'"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Failed for s='0', old='0', new='00': expected '00' but got '0'
Falsifying example: test_replace_matches_python(
    strings=['0'],
    old='0',
    new='00',
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

print("Testing numpy.strings.replace() bug:")
print("="*50)
print()

test_cases = [
    ('0', '0', '00'),
    ('a', 'a', 'aa'),
    ('hello', 'hello', 'hellohello'),
    ('hello', 'l', 'll'),
]

for s, old, new in test_cases:
    arr = np.array([s])
    py_result = s.replace(old, new)
    np_result = nps.replace(arr, old, new)[0]
    match = 'PASS' if py_result == np_result else 'FAIL'
    print(f"{match}: replace('{s}', '{old}', '{new}')")
    print(f"  Expected (Python): '{py_result}'")
    print(f"  Actual   (NumPy):  '{np_result}'")
    print()
```

<details>

<summary>
Bug demonstration showing silent failure for whole-string replacements
</summary>
```
Testing numpy.strings.replace() bug:
==================================================

FAIL: replace('0', '0', '00')
  Expected (Python): '00'
  Actual   (NumPy):  '0'

FAIL: replace('a', 'a', 'aa')
  Expected (Python): 'aa'
  Actual   (NumPy):  'a'

FAIL: replace('hello', 'hello', 'hellohello')
  Expected (Python): 'hellohello'
  Actual   (NumPy):  'hello'

PASS: replace('hello', 'l', 'll')
  Expected (Python): 'hellllo'
  Actual   (NumPy):  'hellllo'

```
</details>

## Why This Is A Bug

The NumPy documentation explicitly states that `numpy.strings.replace` performs "element-wise string replacement" and matches Python's `str.replace` method functionality. This bug violates that contract in a specific but important case:

1. **Silent failure pattern**: When `old == s` (the substring to replace equals the entire string) AND `len(new) > len(old)` (the replacement is longer), NumPy silently returns the original string unchanged instead of performing the replacement.

2. **Inconsistent behavior**: The function works correctly for partial replacements that expand the string (e.g., replacing 'l' with 'll' in 'hello' correctly produces 'hellllo'), but fails for whole-string replacements.

3. **No error or warning**: The operation appears to succeed but produces incorrect results without any indication of failure, leading to potential data corruption.

4. **Documentation violation**: The function explicitly promises to match Python's `str.replace()` behavior, which has no such limitation.

## Relevant Context

The bug appears to be in the buffer size calculation in `numpy/_core/strings.py` at lines 1363-1367:

```python
buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))
out_dtype = f"{arr.dtype.char}{buffersizes.max()}"
out = np.empty_like(arr, shape=buffersizes.shape, dtype=out_dtype)
return _replace(arr, old, new, counts, out=out)
```

When the entire string is replaced, the calculation `str_len(arr) + counts * (str_len(new) - str_len(old))` correctly computes the needed buffer size. However, it appears the underlying `_replace` ufunc may have a separate check or limitation that prevents it from replacing when the pattern equals the entire string and would expand beyond the original size.

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.strings.replace.html

## Proposed Fix

The issue likely lies in the underlying C implementation of the `_replace` ufunc. A proper fix would require:

1. Investigating the `_replace` ufunc implementation to understand why it fails to perform the replacement when `old == s` and `len(new) > len(old)`
2. Removing any artificial restrictions on whole-string replacements
3. Ensuring the output buffer is properly utilized for the expanded string

As a workaround, users can manually pre-allocate arrays with sufficient size or use list comprehensions with Python's `str.replace()` instead of relying on NumPy's vectorized version for cases involving whole-string replacements.