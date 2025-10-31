# Bug Report: numpy.char.partition and rpartition Lose Null Bytes

**Target**: `numpy.char.partition`, `numpy.char.rpartition`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.partition()` and `numpy.char.rpartition()` strip null bytes (`\x00`) from their results, causing silent data corruption.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st


@given(st.lists(st.text(), min_size=1), st.text(min_size=1))
def test_partition_matches_python(strings, sep):
    arr = np.array(strings)
    numpy_result = numpy.char.partition(arr, sep)

    for i in range(len(strings)):
        python_result = strings[i].partition(sep)
        assert tuple(numpy_result[i]) == python_result
```

**Failing input**: `strings=['\x00'], sep='0'`

## Reproducing the Bug

```python
import numpy as np
import numpy.char

test_string = '\x00'
sep = '0'

arr = np.array([test_string])

python_result = test_string.partition(sep)
numpy_result = numpy.char.partition(arr, sep)

print(f"Python partition: {python_result}")
print(f"NumPy partition:  {tuple(numpy_result[0])}")

python_rpart = test_string.rpartition(sep)
numpy_rpart = numpy.char.rpartition(arr, sep)

print(f"\nPython rpartition: {python_rpart}")
print(f"NumPy rpartition:  {tuple(numpy_rpart[0])}")
```

Output:
```
Python partition: ('\x00', '', '')
NumPy partition:  ('', '', '')

Python rpartition: ('', '', '\x00')
NumPy rpartition:  ('', '', '')
```

## Why This Is A Bug

1. **Silent data corruption**: Null bytes are valid string content but are silently removed
2. **Violates documented contract**: Claims to call `str.partition/rpartition` but doesn't match Python's behavior
3. **Breaks real use cases**: Binary data encoded as strings, network protocols, file formats all use null bytes

## Fix

The issue appears to be C string handling treating `\x00` as a string terminator. The fix requires:

1. Use length-aware string operations instead of null-terminated C strings
2. Ensure partition results preserve all characters including null bytes
3. Add tests to verify null byte handling throughout the string manipulation functions