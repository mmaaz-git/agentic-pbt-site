# Bug Report: numpy.strings.swapcase Unicode Character Handling

**Target**: `numpy.strings.swapcase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.strings.swapcase` incorrectly handles special Unicode characters that have multi-character case mappings, producing different results than Python's `str.swapcase` despite documentation claiming it "calls str.swapcase element-wise."

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st

@given(st.text())
def test_swapcase_involution(s):
    arr = np.array([s])
    result = np.strings.swapcase(np.strings.swapcase(arr))
    assert np.array_equal(result, arr)
```

**Failing input**: `'ß'`

## Reproducing the Bug

```python
import numpy as np

s = 'ß'
arr = np.array([s])

once = np.strings.swapcase(arr)
print(f"NumPy:  {s!r} → {once[0]!r}")
print(f"Python: {s!r} → {s.swapcase()!r}")
```

**Output**:
```
NumPy:  'ß' → 'S'
Python: 'ß' → 'SS'
```

Additional failing cases:
- Turkish capital İ: NumPy gives 'i', Python gives 'i̇' (with combining dot)

## Why This Is A Bug

The docstring at line 1184 explicitly states: "Calls :meth:`str.swapcase` element-wise." However, NumPy's implementation produces different results than Python's `str.swapcase` for Unicode characters with special case mappings.

The German ß (eszett) has an uppercase form of 'SS' (two characters), but NumPy only produces a single 'S'. This violates the documented contract and produces incorrect results for German text.

## Fix

The issue is that `numpy.strings.swapcase` calls the underlying `_vec_string` function (line 1214), which likely doesn't properly handle Unicode case folding rules. The fix would require ensuring that the C-level string operations in NumPy's ufunc implementation correctly handle:

1. Characters that expand to multiple characters when case-swapped (like ß → SS)
2. Characters with combining marks (like İ → i̇)

A proper fix would need to update the underlying ufunc implementation to match Python's Unicode case mapping behavior, which may require significant changes to handle variable-length output from swapcase operations.