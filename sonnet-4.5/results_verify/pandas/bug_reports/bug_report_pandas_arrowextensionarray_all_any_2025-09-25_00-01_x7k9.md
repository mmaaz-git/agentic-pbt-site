# Bug Report: ArrowExtensionArray.all() and .any() Fail on Null-Type Arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.all` and `pandas.core.arrays.arrow.ArrowExtensionArray.any`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Calling `all()` or `any()` on an ArrowExtensionArray with dtype null (containing only None values) raises a `TypeError` instead of returning the documented behavior. According to the docstrings, when `skipna=True` and the entire array is NA, `all()` should return True and `any()` should return False (as for an empty array).

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arrow_int_array = st.lists(
    st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()),
    min_size=0,
    max_size=100
).map(lambda x: ArrowExtensionArray(pa.array(x)))

@given(arrow_int_array)
@settings(max_examples=200)
def test_all_true_implies_any_true(arr):
    assume(len(arr) > 0)

    if arr.all(skipna=True):
        assert arr.any(skipna=True), "If all() is True, any() should also be True"
```

**Failing input**: `arr = ArrowExtensionArray(pa.array([None]))`

## Reproducing the Bug

```python
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arr = ArrowExtensionArray(pa.array([None]))

result_all = arr.all(skipna=True)
```

**Output**:
```
TypeError: 'ArrowExtensionArray' with dtype null[pyarrow] does not support reduction 'all' with pyarrow version 20.0.0. 'all' may be supported by upgrading pyarrow.
```

Similarly for `any()`:
```python
result_any = arr.any(skipna=True)
```

**Output**:
```
TypeError: 'ArrowExtensionArray' with dtype null[pyarrow] does not support reduction 'any' with pyarrow version 20.0.0. 'any' may be supported by upgrading pyarrow.
```

## Why This Is A Bug

The docstrings for both methods explicitly document the expected behavior when the entire array is NA:

**From `all()` docstring:**
> "If the entire array is NA and `skipna` is True, then the result will be True, as for an empty array."

**From `any()` docstring:**
> "If the entire array is NA and `skipna` is True, then the result will be False, as for an empty array."

The current implementation violates this documented contract by raising an exception instead of returning the specified values.

This is realistic because users may have arrays that are entirely NA during data processing (e.g., filtering, initialization), and calling reduction operations should work consistently.

## Fix

The `_reduce_pyarrow` method should handle null-type arrays as a special case before attempting PyArrow reductions:

```diff
def _reduce_pyarrow(self, name: str, *, skipna: bool = True, **kwargs) -> pa.Scalar:
    pa_type = self._pa_array.type

+    # Handle null-type arrays specially
+    if pa.types.is_null(pa_type):
+        if skipna:
+            # When skipna=True and all values are NA, follow documented behavior
+            if name == "all":
+                return pa.scalar(True, type=pa.bool_())
+            elif name == "any":
+                return pa.scalar(False, type=pa.bool_())
+            # For other reductions on all-NA with skipna=True, return NA
+            return pa.scalar(None, type=pa.null())
+        else:
+            # When skipna=False and any NA is present, return NA
+            return pa.scalar(None, type=pa.null())
+
    data_to_reduce = self._pa_array
    # ... rest of implementation
```