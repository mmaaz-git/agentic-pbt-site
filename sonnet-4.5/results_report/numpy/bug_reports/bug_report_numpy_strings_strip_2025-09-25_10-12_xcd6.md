# Bug Report: numpy.strings strip() Incorrectly Strips Null Bytes

**Target**: `numpy.strings.strip()`, `numpy.strings.lstrip()`, `numpy.strings.rstrip()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.strings.strip()` (and its variants `lstrip()` and `rstrip()`) incorrectly strip null bytes (`\x00`) from strings, while Python's `str.strip()` does not. This violates the documented behavior that these functions should match Python's string methods.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st


@given(st.lists(st.text(), min_size=1, max_size=10))
def test_operations_on_arrays(strings):
    arr = np.array(strings)
    result = nps.strip(arr)
    assert len(result) == len(arr)
    for i, (original, stripped) in enumerate(zip(strings, result)):
        assert stripped == original.strip()
```

**Failing input**: `['\x00']`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

s = '\x00'
python_result = s.strip()
numpy_result = nps.strip(np.array([s]))[0]

print(f"Python str.strip(): {repr(python_result)}")
print(f"numpy.strings.strip(): {repr(numpy_result)}")

assert python_result == '\x00'
assert numpy_result == ''
```

## Why This Is A Bug

Python's `str.strip()` only removes characters classified as whitespace according to Unicode. The null byte (`\x00`) is **not** whitespace, so Python correctly preserves it. However, `numpy.strings.strip()` incorrectly removes it, violating the expected compatibility with Python's string methods.

This affects all three variants: `strip()`, `lstrip()`, and `rstrip()`.

## Fix

The implementation should match Python's definition of whitespace characters. The bug likely stems from using a C-style string terminator check or an incorrect whitespace character set. The fix requires updating the internal whitespace detection to match Python's Unicode whitespace definition exactly.