# Bug Report: IntegerArray fillna/setitem Overflow Validation

**Target**: `pandas.arrays.IntegerArray.fillna` and `pandas.arrays.IntegerArray.__setitem__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

IntegerArray's `fillna()` and `__setitem__()` methods crash with an obscure OverflowError when given integer values outside the int64 range, instead of raising a clear validation error like other methods do.

## Property-Based Test

```python
import numpy as np
import pandas.arrays as pa
from hypothesis import given, strategies as st, assume


@st.composite
def integer_array_with_na(draw):
    size = draw(st.integers(min_value=1, max_value=20))
    values = draw(st.lists(st.integers(min_value=-100, max_value=100), min_size=size, max_size=size))
    mask = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    assume(any(mask))
    return pa.IntegerArray(np.array(values, dtype='int64'), np.array(mask, dtype='bool'))


@given(integer_array_with_na(), st.integers())
def test_fillna_accepts_any_integer(arr, fill_value):
    result = arr.fillna(fill_value)
    assert not result.isna().any()
```

**Failing input**: `fill_value=9_223_372_036_854_775_808` (2^63, one more than max int64)

## Reproducing the Bug

```python
import numpy as np
import pandas.arrays as pa

arr = pa.IntegerArray(np.array([1, 2, 3]), np.array([False, True, False]))

overflow_value = 2**63

arr.fillna(overflow_value)

arr_copy = arr.copy()
arr_copy[1] = overflow_value
```

Both operations raise:
```
OverflowError: Python int too large to convert to C long
```

## Why This Is A Bug

1. **Inconsistent error handling**: The `insert()` method properly validates and raises a clear TypeError: `"cannot safely cast non-equivalent uint64 to int64"`, but `fillna()` and `__setitem__()` crash with an obscure OverflowError.

2. **Poor error message**: The error "Python int too large to convert to C long" is cryptic and doesn't indicate that the value is out of range for the Int64 dtype.

3. **Type validation works correctly**: The code properly validates types (rejecting floats and strings with clear TypeErrors), so it should also validate value ranges.

## Fix

The bug is in `pandas/core/arrays/masked.py` in the `__setitem__` method around line 316. The fix should add range validation similar to what `insert()` does via `_safe_cast()`:

```diff
--- a/pandas/core/arrays/masked.py
+++ b/pandas/core/arrays/masked.py
@@ -313,7 +313,14 @@ class BaseMaskedArray(ExtensionArray):
             if is_valid_na_for_dtype(value, self.dtype):
                 self._mask[key] = True
             else:
                 value = self._validate_setitem_value(value)
-                self._data[key] = value
+                try:
+                    self._data[key] = value
+                except OverflowError as err:
+                    raise TypeError(
+                        f"Cannot assign value {value} to {self.dtype}. "
+                        f"Value is outside the valid range for {self.dtype.name}."
+                    ) from err
```

Alternatively, add validation in `_validate_setitem_value()` to check if the value is within the valid range for the dtype before attempting assignment.