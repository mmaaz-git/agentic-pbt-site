# Bug Report: ListAccessor.__getitem__ Crashes on Empty Lists

**Target**: `pandas.core.arrays.arrow.accessors.ListAccessor.__getitem__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When accessing elements from a Series containing PyArrow list arrays, the `ListAccessor.__getitem__` method crashes with an `ArrowInvalid` error when the Series contains any empty lists, even though the documentation doesn't warn about this limitation and other list accessor methods handle empty lists correctly.

## Property-Based Test

```python
import pandas as pd
import pyarrow as pa
from hypothesis import given, strategies as st, settings

@st.composite
def arrow_list_arrays(draw):
    lists = draw(st.lists(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=0, max_size=10),
        min_size=1,
        max_size=20
    ))
    return pd.Series(lists, dtype=pd.ArrowDtype(pa.list_(pa.int64())))

@given(arrow_list_arrays())
@settings(max_examples=200)
def test_list_getitem_handles_empty_lists(series):
    # This should not crash - it should return NA for empty lists
    result = series.list[0]
    assert len(result) == len(series)
```

**Failing input**: `series = pd.Series([[], [0]], dtype=pd.ArrowDtype(pa.list_(pa.int64())))`

## Reproducing the Bug

```python
import pandas as pd
import pyarrow as pa

lists = [[], [0]]
series = pd.Series(lists, dtype=pd.ArrowDtype(pa.list_(pa.int64())))

result = series.list[0]
```

**Output:**
```
ArrowInvalid: Index 0 is out of bounds: should be in [0, 0)
```

## Why This Is A Bug

1. **Undocumented limitation**: The docstring for `ListAccessor.__getitem__` (lines 118-147 in accessors.py) doesn't mention that it will crash on empty lists.

2. **Inconsistent behavior**: Other `ListAccessor` methods like `len()` and `flatten()` handle empty lists correctly:
   ```python
   series.list.len()      # Works fine: returns [0, 1]
   series.list.flatten()  # Works fine: returns [0]
   series.list[0]         # Crashes!
   ```

3. **Reasonable expectation**: Users would expect `series.list[0]` to return `NA` (missing value) for empty lists, similar to how pandas handles out-of-bounds access in other contexts.

4. **Real-world impact**: Real datasets commonly contain mixed empty and non-empty lists. This crash makes the accessor unusable for such common scenarios.

## Fix

The bug is in `/pandas/core/arrays/arrow/accessors.py` at line 155, where `pc.list_element()` is called without checking list lengths first.

```diff
--- a/pandas/core/arrays/arrow/accessors.py
+++ b/pandas/core/arrays/arrow/accessors.py
@@ -148,9 +148,22 @@ class ListAccessor(ArrowAccessor):
         from pandas import Series

         if isinstance(key, int):
-            # TODO: Support negative key but pyarrow does not allow
-            # element index to be an array.
-            # if key < 0:
-            #     key = pc.add(key, pc.list_value_length(self._pa_array))
-            element = pc.list_element(self._pa_array, key)
+            # Check if any lists are too short to contain the requested element
+            lengths = pc.list_value_length(self._pa_array)
+            min_required_length = key + 1 if key >= 0 else abs(key)
+            all_long_enough = pc.all(pc.greater_equal(lengths, min_required_length)).as_py()
+
+            if all_long_enough:
+                # Fast path: all lists are long enough, use pyarrow directly
+                element = pc.list_element(self._pa_array, key)
+            else:
+                # Slow path: some lists are too short, build result manually
+                result_values = []
+                for lst in self._pa_array.to_pylist():
+                    if lst is not None and len(lst) > key:
+                        result_values.append(lst[key])
+                    else:
+                        result_values.append(None)
+                element = pa.array(result_values, type=self._pa_array.type.value_type)
+
             return Series(element, dtype=ArrowDtype(element.type))
```