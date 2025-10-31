# Bug Report: pandas.core.strings.accessor.cat_core Null Byte Separator Bug

**Target**: `pandas.core.strings.accessor.cat_core`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cat_core` function incorrectly handles null byte (`\x00`) separators, dropping them entirely during string concatenation and producing incorrect results.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, assume, strategies as st
from pandas.core.strings.accessor import cat_core


@given(
    st.lists(st.text(), min_size=1, max_size=5),
    st.lists(st.text(), min_size=1, max_size=5),
    st.text(max_size=5)
)
@settings(max_examples=1000)
def test_cat_core_correctness(col1, col2, sep):
    assume(len(col1) == len(col2))

    arr1 = np.array(col1, dtype=object)
    arr2 = np.array(col2, dtype=object)

    result = cat_core([arr1, arr2], sep)

    for i in range(len(col1)):
        expected = col1[i] + sep + col2[i]
        assert result[i] == expected
```

**Failing input**: `col1=[''], col2=[''], sep='\x00'`

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas.core.strings.accessor import cat_core

arr1 = np.array(['a'], dtype=object)
arr2 = np.array(['b'], dtype=object)
result = cat_core([arr1, arr2], '\x00')

print(f"Expected: {'a' + '\x00' + 'b'!r}")
print(f"Got: {result[0]!r}")
assert result[0] == 'a\x00b', f"Bug: separator was dropped"

s1 = pd.Series(['a', 'hello', ''])
s2 = pd.Series(['b', 'world', ''])
result_api = s1.str.cat(s2, sep='\x00')

for i in range(len(s1)):
    expected = s1.iloc[i] + '\x00' + s2.iloc[i]
    assert result_api.iloc[i] == expected, f"Bug in user-facing API at index {i}"
```

## Why This Is A Bug

The `cat_core` function is documented to concatenate arrays with a separator. However, when the separator is a null byte (`\x00`), it gets dropped entirely. This violates the basic contract that joining strings with a separator should include that separator between elements.

The root cause is in these lines:

```python
list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
list_with_sep[::2] = list_of_columns
arr_with_sep = np.asarray(list_with_sep, dtype=object)
return np.sum(arr_with_sep, axis=0)
```

When `np.asarray` creates an array from a list containing both numpy arrays and scalar strings, and then `np.sum` is called with `axis=0`, the scalar string separators are not properly included in the concatenation. This happens specifically with null bytes (and potentially other characters that have special meaning in C strings).

## Fix

The fix is to convert separator strings into arrays of the same shape as the column arrays before concatenation:

```diff
--- a/pandas/core/strings/accessor.py
+++ b/pandas/core/strings/accessor.py
@@ -100,9 +100,11 @@ def cat_core(list_of_columns: list, sep: str):
     if sep == "":
         arr_of_cols = np.asarray(list_of_columns, dtype=object)
         return np.sum(arr_of_cols, axis=0)
-    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
-    list_with_sep[::2] = list_of_columns
-    arr_with_sep = np.asarray(list_with_sep, dtype=object)
-    return np.sum(arr_with_sep, axis=0)
+
+    n = len(list_of_columns[0])
+    sep_arrays = [np.array([sep] * n, dtype=object) for _ in range(len(list_of_columns) - 1)]
+    interleaved = [list_of_columns[0]]
+    for sep_arr, col in zip(sep_arrays, list_of_columns[1:]):
+        interleaved.extend([sep_arr, col])
+    return np.sum(interleaved, axis=0)
```
