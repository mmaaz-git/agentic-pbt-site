# Bug Report: pandas.core.arrays.arrow ListAccessor.__getitem__ Crashes on Variable-Length Lists

**Target**: `pandas.core.arrays.arrow.accessors.ListAccessor.__getitem__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When using `ListAccessor.__getitem__()` with an integer index on a Series containing lists of different lengths, the method raises an `ArrowInvalid` exception if the index is out of bounds for any list, even though the docstring suggests it should "access from each list" independently.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import pyarrow as pa


@settings(max_examples=500)
@given(
    st.lists(
        st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=10),
        min_size=1,
        max_size=20
    ),
    st.integers(min_value=0, max_value=9)
)
def test_list_accessor_getitem_returns_correct_element(lists_of_ints, index):
    s = pd.Series(lists_of_ints, dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    result = s.list[index]

    expected = [lst[index] if index < len(lst) else None for lst in lists_of_ints]

    for i, (res, exp) in enumerate(zip(result, expected)):
        if exp is None:
            assert pd.isna(res)
        else:
            assert res == exp
```

**Failing input**: `lists_of_ints=[[0]], index=1`

## Reproducing the Bug

```python
import pandas as pd
import pyarrow as pa

s = pd.Series(
    [[1, 2, 3], [4]],
    dtype=pd.ArrowDtype(pa.list_(pa.int64()))
)

print(s.list[0])

s.list[1]
```

**Output:**
```
pyarrow.lib.ArrowInvalid: Index 1 is out of bounds: should be in [0, 1)
```

## Why This Is A Bug

1. **Inconsistent with pandas philosophy**: Pandas typically returns NA/null for out-of-bounds access rather than raising exceptions
2. **Undocumented limitation**: The docstring for `__getitem__` says "Index or slice of indices to access from each list" but doesn't mention that the index must be valid for ALL lists
3. **Unintuitive behavior**: Users naturally expect that when accessing lists of different lengths, shorter lists would return NA for out-of-bounds indices, similar to how `Series.str.get()` works
4. **Poor error message**: The error "Index 1 is out of bounds: should be in [0, 1)" doesn't clearly explain that it's failing because ONE of the lists is too short

## Fix

The underlying issue is that `pyarrow.compute.list_element()` requires the index to be valid for all lists. Pandas should handle this gracefully by:

1. Checking list lengths before calling `pc.list_element()`
2. For lists shorter than the index, returning NA
3. Only calling `pc.list_element()` on lists that have the index

A potential fix:

```diff
--- a/pandas/core/arrays/arrow/accessors.py
+++ b/pandas/core/arrays/arrow/accessors.py
@@ -148,9 +148,23 @@ class ListAccessor(ArrowAccessor):
         from pandas import Series

         if isinstance(key, int):
-            # TODO: Support negative key but pyarrow does not allow
-            # element index to be an array.
-            element = pc.list_element(self._pa_array, key)
+            if key < 0:
+                raise NotImplementedError("Negative indexing is not yet supported")
+
+            # Handle variable-length lists by checking lengths first
+            lengths = pc.list_value_length(self._pa_array)
+
+            # Check if any list is too short
+            min_length = pc.min(lengths).as_py()
+            if min_length <= key:
+                # Some lists don't have this index - need element-wise handling
+                # For now, provide a clearer error message
+                raise IndexError(
+                    f"Index {key} is out of bounds for lists with minimum length {min_length}. "
+                    f"Variable-length list indexing with out-of-bounds handling is not yet supported."
+                )
+
+            element = pc.list_element(self._pa_array, key)
             return Series(element, dtype=ArrowDtype(element.type))
         elif isinstance(key, slice):
```

**Note**: The above fix improves the error message but doesn't fully solve the problem. A complete fix would require implementing element-wise null handling for out-of-bounds access, which is more complex and would need to construct a result array with nulls for out-of-bounds indices.