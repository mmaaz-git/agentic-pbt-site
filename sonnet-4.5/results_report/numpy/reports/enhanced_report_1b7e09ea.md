# Bug Report: numpy.char.partition/rpartition Incorrectly Rejects Null Byte Separator

**Target**: `numpy.char.partition`, `numpy.char.rpartition`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.char.partition()` and `numpy.char.rpartition()` incorrectly reject null bytes (`\x00`) as separators, raising `ValueError: empty separator`, while Python's standard `str.partition()` and `str.rpartition()` accept null bytes as valid single-character separators. This violates the documented contract that these functions "call str.partition element-wise".

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=20))
@settings(max_examples=1000)
def test_partition_matches_python_behavior(s):
    sep = '\x00'

    try:
        py_result = s.partition(sep)
    except ValueError:
        py_result = None

    try:
        np_result = char.partition(s, sep)
        if hasattr(np_result, 'shape') and np_result.shape == ():
            np_result = np_result.item()
        np_result = tuple(str(p) for p in np_result)
    except ValueError:
        np_result = None

    if (py_result is None) != (np_result is None):
        raise AssertionError(
            f"partition('{s}', {repr(sep)}) behavior differs: "
            f"Python: {py_result}, NumPy: {np_result}"
        )

if __name__ == "__main__":
    test_partition_matches_python_behavior()
```

<details>

<summary>
**Failing input**: `s=''` (or any string value)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 29, in <module>
    test_partition_matches_python_behavior()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 5, in test_partition_matches_python_behavior
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 23, in test_partition_matches_python_behavior
    raise AssertionError(
    ...<2 lines>...
    )
AssertionError: partition('', '\x00') behavior differs: Python: ('', '', ''), NumPy: None
Falsifying example: test_partition_matches_python_behavior(
    s='',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy.char as char

s = 'test'
sep = '\x00'

print("Python str.partition:")
print(s.partition(sep))

print("\nnumpy.char.partition:")
try:
    result = char.partition(s, sep)
    print(result)
except ValueError as e:
    print(f"ValueError: {e}")

print("\nPython str.rpartition:")
print(s.rpartition(sep))

print("\nnumpy.char.rpartition:")
try:
    result = char.rpartition(s, sep)
    print(result)
except ValueError as e:
    print(f"ValueError: {e}")
```

<details>

<summary>
ValueError raised for both numpy.char functions with null byte separator
</summary>
```
Python str.partition:
('test', '', '')

numpy.char.partition:
ValueError: empty separator

Python str.rpartition:
('', '', 'test')

numpy.char.rpartition:
ValueError: empty separator
```
</details>

## Why This Is A Bug

This is a clear contract violation that breaks compatibility with Python's standard library:

1. **Documentation explicitly promises Python compatibility**: The numpy.char.partition docstring states "Calls :meth:`str.partition` element-wise", establishing a contract to match Python's str.partition behavior.

2. **Incorrect validation logic**: The null byte `\x00` is a valid single-character string in Python (len('\x00') == 1), but NumPy's implementation incorrectly treats it as "empty". This suggests the validation code is using a C-style null-termination check or boolean test instead of checking the actual string length.

3. **Behavioral inconsistency**:
   - Python's `str.partition('\x00')` correctly treats null bytes as valid separators and returns the expected 3-tuple
   - NumPy raises `ValueError: empty separator` for the same input
   - Both correctly raise ValueError for truly empty separators ('')

4. **Real-world impact**: Null bytes are commonly encountered in:
   - Binary protocol parsing
   - C string interoperability (null-terminated strings)
   - File format parsing
   - Network packet processing

   Users working with such data expect NumPy to handle null bytes correctly as Python does.

5. **Multiple functions affected**: Both `partition` and `rpartition` exhibit the same bug, suggesting a shared validation routine with the incorrect logic.

## Relevant Context

The error occurs in the low-level implementation (`_partition_index` function) which appears to be compiled code. The traceback shows:

```
File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/strings.py", line 1618, in partition
    return _partition_index(a, sep, pos, out=(out["f0"], out["f1"], out["f2"]))
ValueError: empty separator
```

This suggests the bug is in NumPy's C/Cython layer where separator validation occurs. The validation likely uses a boolean or null-termination check that incorrectly treats `\x00` as falsy/empty.

Documentation references:
- numpy.char.partition: https://numpy.org/doc/stable/reference/generated/numpy.char.partition.html
- Python str.partition: https://docs.python.org/3/library/stdtypes.html#str.partition

## Proposed Fix

The fix requires modifying the separator validation logic in NumPy's compiled code. The validation should explicitly check the string length rather than using boolean or C-style null checks:

```diff
# Conceptual fix in the C/Cython layer
- if (!sep || *sep == '\0')  // Incorrect: treats '\x00' as empty
+ if (sep_length == 0)        // Correct: only empty string is invalid
     raise ValueError("empty separator")
```

Since the actual implementation is in compiled code, the exact fix would need to be applied in NumPy's string operations C extension, likely in the `_partition_index` function or its validation routine. The fix should ensure that only truly empty separators (length 0) raise an error, while single-character separators including null bytes are accepted.