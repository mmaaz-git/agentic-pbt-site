# Bug Report: numpy.strings.slice Negative Start with None Stop

**Target**: `numpy.strings.slice`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.slice` incorrectly handles negative `start` indices when `stop=None`. Instead of slicing from the negative index to the end `[start:]`, it incorrectly slices from the beginning to the negative index `[:start]`, returning the wrong portion of the string.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10),
       st.integers(min_value=-20, max_value=-1))
def test_slice_negative_start_none_stop(strings, start):
    arr = np.array(strings)
    result = nps.slice(arr, start, None)

    for i, s in enumerate(strings):
        expected = s[start:]
        assert result[i] == expected
```

**Failing input**: `strings=['hello'], start=-3`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

arr = np.array(['hello'])

result = nps.slice(arr, -3, None)
expected = 'hello'[-3:]

print(f"Input: 'hello'")
print(f"numpy.strings.slice(arr, -3, None): '{result[0]}'")
print(f"Expected 'hello'[-3:]: '{expected}'")

result = nps.slice(arr, -1, None)
print(f"\nnumpy.strings.slice(arr, -1, None): '{result[0]}'")
print(f"Expected 'hello'[-1:]: '{arr[0][-1:]}'")

result = nps.slice(arr, -5, None)
print(f"\nnumpy.strings.slice(arr, -5, None): '{result[0]}'")
print(f"Expected 'hello'[-5:]: '{arr[0][-5:]}'")
```

Output:
```
Input: 'hello'
numpy.strings.slice(arr, -3, None): 'he'
Expected 'hello'[-3:]: 'llo'

numpy.strings.slice(arr, -1, None): 'hell'
Expected 'hello'[-1:]: 'o'

numpy.strings.slice(arr, -5, None): ''
Expected 'hello'[-5:]: 'hello'
```

## Why This Is A Bug

1. **Incorrect results**: The function returns the wrong substring, effectively reversing the meaning of negative indices when `stop=None`
2. **Violates Python slicing semantics**: `s[start:None]` should be equivalent to `s[start:]`, but numpy.strings.slice behaves like `s[:start]`
3. **Inconsistent behavior**: When both `start` and `stop` are negative (e.g., `slice(arr, -5, -2)`), it works correctly. The bug only occurs when `stop=None`
4. **Documentation claims it works**: The docstring shows examples with negative indices that should work, though they use the single-argument form which has different semantics

## Fix

The bug appears to be in the parameter interpretation logic. When `stop=None` and `start` is negative, the function incorrectly swaps their roles. The fix would involve correcting the slice boundary calculation:

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -slice_function
-    # Current logic incorrectly handles negative start with None stop
+    # When stop is None and start is negative, ensure start is used as the start index
+    # Likely the issue is in how None is converted or how the slice is constructed
```

The exact fix location would require examining the source implementation of the `slice` function in `/numpy/_core/strings.py` or the underlying C implementation. The bug is likely in the logic that converts None to actual slice boundaries or in how the slice parameters are passed to the underlying string slicing operation.