# Bug Report: ArrowExtensionArray.insert() Fails on Null-Type Arrays

**Target**: `pandas.core.arrays.arrow.ArrowExtensionArray.insert`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Inserting a non-None value into an ArrowExtensionArray with dtype null (i.e., containing only None values) raises `ArrowNotImplementedError` instead of successfully inserting the value.

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

@given(arrow_int_array, st.integers(min_value=0, max_value=50), st.integers(min_value=-100, max_value=100))
@settings(max_examples=200)
def test_insert_delete_inverse(arr, loc_int, item):
    assume(len(arr) > 0)
    loc = loc_int % (len(arr) + 1)

    inserted = arr.insert(loc, item)
    deleted = inserted.delete(loc)

    assert arr.equals(deleted), "delete(insert(arr, loc, item), loc) should equal arr"
```

**Failing input**: `arr = ArrowExtensionArray(pa.array([None]))`, `loc = 0`, `item = 0`

## Reproducing the Bug

```python
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arr = ArrowExtensionArray(pa.array([None]))
result = arr.insert(0, 42)
```

**Output**:
```
pyarrow.lib.ArrowNotImplementedError: Unsupported cast from int64 to null using function cast_null
```

## Why This Is A Bug

The `insert()` method is documented to insert an item at a given position. When an array contains only None values, it has PyArrow type `null`. Attempting to insert a non-None value into this array should either:
1. Succeed by converting the array to an appropriate type (e.g., int64), or
2. Provide a clear error message explaining the limitation

Instead, it raises a low-level PyArrow exception about type casting that is not user-friendly and violates the expected behavior of insert().

This is a realistic use case: users may start with an empty or all-None array and progressively add values.

## Fix

The issue occurs in the `insert()` method's type handling. When the current array has type `null` and a non-None value is being inserted, the method should detect this case and convert the array to an appropriate type before insertion.

A potential fix would be to check if the array has type `null` and handle it specially:

```diff
def insert(self, loc: int, item) -> Self:
+    # Handle insertion into null-type arrays
+    if pa.types.is_null(self._pa_array.type) and item is not None:
+        # Infer type from the item being inserted
+        item_arr = pa.array([item])
+        new_type = item_arr.type
+        # Convert null array to appropriate type
+        converted = pa.array([None] * len(self), type=new_type)
+        temp_arr = type(self)(converted)
+        return temp_arr.insert(loc, item)
+
    item_arr = type(self)._from_sequence([item], dtype=self.dtype)
    # ... rest of implementation
```

Alternatively, the method could raise a more informative error message explaining that insertion into null-type arrays requires explicit type specification.