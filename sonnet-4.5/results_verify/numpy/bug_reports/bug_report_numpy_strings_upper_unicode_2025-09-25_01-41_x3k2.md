# Bug Report: numpy.strings.upper Unicode Character Expansion

**Target**: `numpy.strings.upper`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.strings.upper` incorrectly handles Unicode characters that expand to multiple characters when uppercased, producing incomplete results that differ from Python's `str.upper` despite documentation claiming element-wise behavior.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st

@given(st.text())
def test_upper_matches_python(s):
    arr = np.array([s])
    numpy_result = np.strings.upper(arr)[0]
    python_result = s.upper()
    assert numpy_result == python_result
```

**Failing input**: `'ß'`

## Reproducing the Bug

```python
import numpy as np

s = 'ß'
arr = np.array([s])
result = np.strings.upper(arr)

print(f"Input:  {s!r}")
print(f"Python: {s.upper()!r}")
print(f"NumPy:  {result[0]!r}")
```

**Output**:
```
Input:  'ß'
Python: 'SS'
NumPy:  'S'
```

## Why This Is A Bug

The docstring at line 1106 states: "Calls :meth:`str.upper` element-wise." However, NumPy's implementation only produces 'S' instead of 'SS' when uppercasing the German ß (eszett). According to Unicode standards and Python's implementation, ß uppercases to 'SS' (two characters), but NumPy truncates this to a single 'S', producing incorrect results for German text.

## Fix

The issue is in line 1135: `return _vec_string(a_arr, a_arr.dtype, 'upper')`. The underlying `_vec_string` function doesn't properly handle Unicode case mappings where a single character expands to multiple characters.

A proper fix would require:
1. Pre-computing the maximum output length by checking for characters that expand during case conversion
2. Allocating sufficient buffer space for expanded characters
3. Updating the underlying C implementation to handle variable-length case mappings

This is similar to how `multiply` (lines 217-226) pre-computes buffer sizes before performing the operation.