# Bug Report: numpy.rec.recarray.field() Unhelpful Error Message

**Target**: `numpy.rec.recarray.field()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `recarray.field()` method raises an unhelpful `IndexError: tuple index out of range` when given an out-of-bounds field index, instead of a clear error indicating which index was invalid and how many fields exist.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.rec
import pytest


@settings(max_examples=500)
@given(
    st.integers(1, 10).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.lists(st.integers(-100, 100), min_size=n, max_size=n),
            st.integers(0, n-1)
        )
    )
)
def test_recarray_field_by_invalid_index(args):
    n, arr, idx = args

    rec_arr = numpy.rec.fromarrays([arr], names='x')

    field = rec_arr.field(idx)
    assert list(field) == arr

    with pytest.raises((IndexError, KeyError)):
        rec_arr.field(n)
```

**Failing input**: `args=(2, [0, 0], 1)` - array with 2 elements, 1 field, trying to access field index 1

## Reproducing the Bug

```python
import numpy.rec

rec_arr = numpy.rec.fromarrays([[0, 0]], names='x')
print(f"Number of fields: {len(rec_arr.dtype.names)}")

rec_arr.field(0)

try:
    rec_arr.field(1)
except IndexError as e:
    print(f"Error: {e}")
```

**Output**:
```
Number of fields: 1
Error: tuple index out of range
```

## Why This Is A Bug

The `field()` method is documented to accept integer indices to access fields by position. When an invalid index is provided, the error message `"tuple index out of range"` is generic and unhelpful - it doesn't tell the user:
1. Which index they attempted to access
2. How many fields are available
3. What the valid index range is

A better error would be: `"Field index 1 out of bounds for array with 1 field"` (similar to how other numpy indexing errors work).

## Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -539,7 +539,11 @@ class recarray(ndarray):

     def field(self, attr, val=None):
         if isinstance(attr, int):
             names = ndarray.__getattribute__(self, 'dtype').names
+            if attr < 0 or attr >= len(names):
+                raise IndexError(
+                    f"Field index {attr} out of bounds for array with {len(names)} field(s)"
+                )
             attr = names[attr]

         fielddict = ndarray.__getattribute__(self, 'dtype').fields