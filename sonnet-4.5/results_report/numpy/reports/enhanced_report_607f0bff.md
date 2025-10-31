# Bug Report: numpy.char.rpartition Null Byte Handling Error

**Target**: `numpy.char.rpartition`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.char.rpartition()` raises `ValueError: empty separator` when given `\x00` (null byte) as separator because NumPy's string conversion treats null bytes as string terminators, converting them to empty strings.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st, assume


@given(st.lists(st.text(), min_size=1), st.text())
def test_rpartition_matches_python(strings, sep):
    assume(len(sep) > 0)
    arr = np.array(strings)

    try:
        numpy_result = numpy.char.rpartition(arr, sep)

        for i in range(len(strings)):
            python_result = strings[i].rpartition(sep)
            assert tuple(numpy_result[i]) == python_result, \
                f"Mismatch for string={repr(strings[i])}, sep={repr(sep)}: " \
                f"NumPy gave {tuple(numpy_result[i])}, Python gave {python_result}"
    except ValueError as e:
        # Check if Python would also raise an error for this separator
        for s in strings:
            try:
                python_result = s.rpartition(sep)
                # If Python succeeded but NumPy failed, this is a bug
                raise AssertionError(
                    f"NumPy raised ValueError: {e} for sep={repr(sep)}, "
                    f"but Python succeeded with result {python_result}"
                )
            except ValueError:
                # Both raised errors, this is fine
                pass


if __name__ == "__main__":
    test_rpartition_matches_python()
```

<details>

<summary>
**Failing input**: `strings=[''], sep='\x00'`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 35, in <module>
  |     test_rpartition_matches_python()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 7, in test_rpartition_matches_python
  |     def test_rpartition_matches_python(strings, sep):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 16, in test_rpartition_matches_python
    |     assert tuple(numpy_result[i]) == python_result, \
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Mismatch for string='0', sep='00': NumPy gave (np.str_(''), np.str_('0'), np.str_('')), Python gave ('', '', '0')
    | Falsifying example: test_rpartition_matches_python(
    |     strings=['0'],
    |     sep='00',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/13/hypo.py:17
    |         /home/npc/pbt/agentic-pbt/worker_/13/hypo.py:19
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 12, in test_rpartition_matches_python
    |     numpy_result = numpy.char.rpartition(arr, sep)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/defchararray.py", line 413, in rpartition
    |     return np.stack(strings_rpartition(a, sep), axis=-1)
    |                     ~~~~~~~~~~~~~~~~~~^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/strings.py", line 1687, in rpartition
    |     return _rpartition_index(
    |         a, sep, pos, out=(out["f0"], out["f1"], out["f2"]))
    | ValueError: empty separator
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 25, in test_rpartition_matches_python
    |     raise AssertionError(
    |     ...<2 lines>...
    |     )
    | AssertionError: NumPy raised ValueError: empty separator for sep='\x00', but Python succeeded with result ('', '', '')
    | Falsifying example: test_rpartition_matches_python(
    |     strings=[''],  # or any other generated value
    |     sep='\x00',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/13/hypo.py:19
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.char

test_string = ''
sep = '\x00'

arr = np.array([test_string])

# Test with Python's built-in rpartition
python_result = test_string.rpartition(sep)
print(f"Python rpartition: {python_result}")

# Test with NumPy's rpartition
try:
    numpy_result = numpy.char.rpartition(arr, sep)
    print(f"NumPy rpartition: {tuple(numpy_result[0])}")
except ValueError as e:
    print(f"NumPy rpartition: ValueError: {e}")
```

<details>

<summary>
ValueError: empty separator
</summary>
```
Python rpartition: ('', '', '')
NumPy rpartition: ValueError: empty separator
```
</details>

## Why This Is A Bug

This violates the expected behavior in multiple ways:

1. **API Contract Violation**: The numpy.char.rpartition documentation states it "Calls str.rpartition element-wise", but Python's `str.rpartition('\x00')` works correctly while NumPy's version raises an error.

2. **String Conversion Issue**: When NumPy converts the null byte `'\x00'` to a NumPy string array, it treats the null byte as a C-style string terminator, converting it to an empty string:
   - Python: `'\x00'` has length 1
   - NumPy: `np.array('\x00')` becomes `array('', dtype='<U1')` with length 0

3. **Inconsistent Behavior**: The function claims to accept any valid separator that Python's str.rpartition accepts, but fails on valid single-character separators like null bytes.

4. **Additional Bug Found**: The Hypothesis test also revealed that `numpy.char.rpartition(['0'], '00')` returns incorrect results compared to Python's implementation (places the string in the wrong position of the tuple).

## Relevant Context

The root cause is that NumPy's string handling uses null-terminated C-style strings internally for Unicode dtypes. When the separator `'\x00'` is converted to a NumPy array with `np.asanyarray(sep)`, it becomes an empty string because null bytes are treated as string terminators.

This can be verified:
```python
>>> import numpy as np
>>> np.array('\x00')
array('', dtype='<U1')  # Null byte becomes empty string
>>> np.strings.str_len(np.array(['\x00']))
array([0])  # Length is 0, not 1
```

The error occurs in the compiled `_rpartition_index` function from `numpy._core.umath`, which checks if the separator is empty and raises the ValueError.

Documentation references:
- numpy.char.rpartition: https://numpy.org/doc/stable/reference/generated/numpy.char.rpartition.html
- Python str.rpartition: https://docs.python.org/3/library/stdtypes.html#str.rpartition

## Proposed Fix

The fix requires handling null bytes specially before string conversion, or using byte arrays internally when null bytes are detected. A high-level approach:

1. Check if the separator contains null bytes before conversion
2. If it does, either:
   - Use byte array processing instead of Unicode strings
   - Special-case null byte handling to preserve it through the conversion
   - Document this as a known limitation for Unicode string arrays

The issue is fundamental to NumPy's string representation and may require architectural changes to fully resolve. A workaround for users is to use byte arrays (`dtype='S'`) instead of Unicode arrays when working with null bytes.