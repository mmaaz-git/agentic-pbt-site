# Bug Report: numpy.char Case Operations Silent Truncation

**Target**: `numpy.char.upper`, `numpy.char.swapcase`, `numpy.char.title`, `numpy.char.capitalize`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Case transformation operations silently truncate results when Unicode characters expand during case conversion (e.g., 'ß' → 'SS'), producing incorrect output without warning.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, settings, strategies as st


@settings(max_examples=500)
@given(st.text(alphabet='ßſ', min_size=1, max_size=10))
def test_upper_truncates_with_inferred_dtype(s):
    arr = np.array([s])
    result = char.upper(arr)

    expected = s.upper()
    actual = str(result[0])

    assert actual == expected, f"upper({s!r}) with dtype {arr.dtype}: expected {expected!r}, got {actual!r}"
```

**Failing input**: `s='ß'`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['ß'])
result = char.upper(arr)

print(f"Input: {arr[0]!r}")
print(f"numpy.char.upper result: {result[0]!r}")
print(f"Python str.upper result: {'ß'.upper()!r}")

assert result[0] == 'SS', f"Expected 'SS', got {result[0]!r}"
```

Output:
```
Input: np.str_('ß')
numpy.char.upper result: np.str_('S')
Python str.upper result: 'SS'
AssertionError: Expected 'SS', got np.str_('S')
```

## Why This Is A Bug

1. The docstring for `numpy.char.upper` states: "Calls str.upper element-wise"
2. Python's `'ß'.upper()` correctly returns `'SS'` (German sharp S uppercases to two capital S's per Unicode standard)
3. numpy.char.upper silently returns `'S'` (truncated) when the array has inferred dtype `<U1`
4. This violates the documented contract and causes silent data corruption

The bug affects all case operations: `upper`, `lower`, `swapcase`, `title`, and `capitalize`. When the inferred dtype is too narrow to hold the transformed result, the output is silently truncated.

## Fix

The issue occurs because numpy infers a narrow dtype (e.g., `<U1` for single-character input) and doesn't automatically resize when case transformations expand the string. Possible fixes:

1. **Auto-resize output array** (preferred): Automatically determine the required output dtype width before performing the operation
2. **Raise warning/error**: Alert users when truncation would occur
3. **Document limitation**: If this is intended behavior, clearly document it in all affected function docstrings

Example fix approach (conceptual):

```diff
def upper(a):
-   # Current: uses input dtype width
+   # Proposed: calculate required output width
+   test_result = np.vectorize(str.upper)(a)
+   max_len = max(len(s) for s in test_result.flat)
+   output_dtype = f'U{max_len}'
+   return np.array([s.upper() for s in a.flat], dtype=output_dtype).reshape(a.shape)
```

Note: The actual implementation is in C, so the fix would need to be applied at that level.