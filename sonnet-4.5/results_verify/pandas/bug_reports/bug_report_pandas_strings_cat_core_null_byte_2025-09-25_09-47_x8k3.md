# Bug Report: pandas.core.strings.accessor.cat_core Null Byte Separator Silently Dropped

**Target**: `pandas.core.strings.accessor.cat_core`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cat_core` function silently drops null byte (`\x00`) characters when used as separators, resulting in incorrect concatenation. All other characters work correctly, but null bytes are completely omitted from the output.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings

import pandas.core.strings.accessor as accessor


@given(
    array_length=st.integers(min_value=1, max_value=10),
    num_arrays=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=500)
def test_cat_core_preserves_separator(array_length, num_arrays):
    sep = '\x00'

    arrays = [
        np.array([f's{i}_{j}' for j in range(array_length)], dtype=object)
        for i in range(num_arrays)
    ]

    result = accessor.cat_core(arrays, sep)

    for i in range(array_length):
        expected = sep.join([arrays[j][i] for j in range(num_arrays)])
        assert result[i] == expected, \
            f"At index {i}: expected {repr(expected)}, got {repr(result[i])}"
```

**Failing input**: `array_length=1, num_arrays=2`

## Reproducing the Bug

```python
import numpy as np
import pandas.core.strings.accessor as accessor

arr1 = np.array(['hello'], dtype=object)
arr2 = np.array(['world'], dtype=object)

result = accessor.cat_core([arr1, arr2], '\x00')

print(f"Result: {repr(result[0])}")
print(f"Expected: {repr('hello\x00world')}")
assert result[0] == 'hello\x00world', "Null byte was silently dropped!"
```

## Why This Is A Bug

1. **Violates function contract**: The function accepts any string as `sep` parameter, with no documented restrictions. Silently dropping characters is unexpected behavior.

2. **Null bytes are valid**: Null bytes are legitimate string characters used in various contexts (binary protocols, null-delimited data, etc.).

3. **Silent data corruption**: The function doesn't raise an error; it just produces incorrect output, making the bug hard to detect.

4. **Inconsistent behavior**: All other special characters (tabs, newlines, etc.) work correctly. Only null bytes are affected.

## Fix

The root cause is that `np.sum` on a mixed object array containing both ndarrays and string scalars doesn't handle null bytes correctly. NumPy silently drops null bytes when adding a string scalar to an array.

The fix is to use explicit element-wise concatenation instead of relying on `np.sum`:

```diff
--- a/pandas/core/strings/accessor.py
+++ b/pandas/core/strings/accessor.py
@@ -XX,XX +XX,XX @@ def cat_core(list_of_columns: list, sep: str):
     if sep == "":
         # no need to interleave sep if it is empty
         arr_of_cols = np.asarray(list_of_columns, dtype=object)
         return np.sum(arr_of_cols, axis=0)
-    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
-    list_with_sep[::2] = list_of_columns
-    arr_with_sep = np.asarray(list_with_sep, dtype=object)
-    return np.sum(arr_with_sep, axis=0)
+
+    # Use explicit concatenation to preserve all characters including null bytes
+    array_length = len(list_of_columns[0])
+    result = np.empty(array_length, dtype=object)
+    for i in range(array_length):
+        result[i] = sep.join([col[i] for col in list_of_columns])
+    return result
```