# Bug Report: xarray.coding.strings.char_to_bytes Shape Loss for Empty Arrays

**Target**: `xarray.coding.strings.char_to_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `char_to_bytes` function incorrectly converts empty 1D character arrays with shape `(0,)` into scalar arrays with shape `()`, violating shape preservation and breaking the inverse property with `bytes_to_char`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from xarray.coding.strings import bytes_to_char, char_to_bytes

@given(
    st.lists(
        st.binary(min_size=1, max_size=100),
        min_size=1,
        max_size=50
    )
)
def test_bytes_to_char_inverse(byte_strings):
    max_len = max(len(s) for s in byte_strings)
    byte_array = np.array(byte_strings, dtype=f'S{max_len}')

    char_array = bytes_to_char(byte_array)
    result = char_to_bytes(char_array)

    np.testing.assert_array_equal(result, byte_array)
```

**Failing input**: Empty list `[]` which creates a byte array with shape `(0,)`

## Reproducing the Bug

```python
import numpy as np
from xarray.coding.strings import char_to_bytes

char_array = np.array([], dtype='S1')
result = char_to_bytes(char_array)

print(f"Input shape: {char_array.shape}")
print(f"Output shape: {result.shape}")

assert result.shape == (0,)
```

**Output:**
```
Input shape: (0,)
Output shape: ()
AssertionError: Shape mismatch: expected (0,), got ()
```

## Why This Is A Bug

1. **Shape preservation violation**: For non-empty arrays, `char_to_bytes` preserves the number of dimensions by removing the last dimension. For example, an array with shape `(3, 5)` becomes `(3,)`. However, an array with shape `(0,)` incorrectly becomes `()` instead of maintaining shape `(0,)`.

2. **Breaks inverse property**: The functions `bytes_to_char` and `char_to_bytes` should be inverse operations, but this bug breaks that property for empty arrays:
   - `bytes_to_char(np.array([], dtype='S5'))` produces shape `(0, 5)`
   - `char_to_bytes(np.array([], dtype='S1'))` produces shape `()` instead of `(0,)`

3. **Inconsistent dimension handling**: The function treats the special case of empty last dimension inconsistently - it should preserve the array as 1D when there's only one dimension left.

## Fix

The bug is in `xarray/coding/strings.py` at lines 207-211:

```python
size = arr.shape[-1]

if not size:
    # can't make an S0 dtype
    return np.zeros(arr.shape[:-1], dtype=np.bytes_)
```

When `arr.shape` is `(0,)`, `arr.shape[:-1]` evaluates to `()`, creating a scalar. The fix should ensure the result maintains at least 1 dimension:

```diff
--- a/xarray/coding/strings.py
+++ b/xarray/coding/strings.py
@@ -208,7 +208,11 @@ def char_to_bytes(arr):

     if not size:
         # can't make an S0 dtype
-        return np.zeros(arr.shape[:-1], dtype=np.bytes_)
+        result_shape = arr.shape[:-1]
+        if len(result_shape) == 0 and arr.ndim == 1:
+            # Preserve 1D shape for empty arrays
+            result_shape = (0,)
+        return np.zeros(result_shape, dtype=np.bytes_)

     if is_chunked_array(arr):
         chunkmanager = get_chunked_array_type(arr)
```