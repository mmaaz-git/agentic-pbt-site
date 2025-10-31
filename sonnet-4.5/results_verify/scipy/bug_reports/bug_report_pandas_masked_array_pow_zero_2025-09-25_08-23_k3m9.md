# Bug Report: pandas.core.arrays NA Power Zero Inconsistency

**Target**: `pandas.core.arrays.IntegerArray.__pow__`, `pandas.core.arrays.FloatingArray.__pow__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Masked arrays (IntegerArray, FloatingArray, BooleanArray) inconsistently handle NA propagation in power operations. `NA ** 0` returns `1` while `NA * 0` returns `NA`, violating the principle that NA should propagate consistently through operations.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings


@settings(max_examples=500)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3) | st.none(), min_size=1, max_size=50))
def test_floatingarray_na_propagation_consistency(values):
    arr = pd.array(values, dtype="Float64")

    pow_zero = arr ** 0
    mul_zero = arr * 0

    for i in range(len(arr)):
        if pd.isna(arr[i]):
            assert pd.isna(mul_zero[i]), f"NA * 0 correctly returns NA"
            assert pd.isna(pow_zero[i]), f"NA ** 0 should return NA but got {pow_zero[i]}"
```

**Failing input**: `values=[None]`

## Reproducing the Bug

```python
import pandas as pd

na_float = pd.array([None], dtype="Float64")
na_int = pd.array([None], dtype="Int64")

print(f"NA ** 0 = {(na_float ** 0)[0]}")
print(f"NA * 0 = {(na_float * 0)[0]}")

print(f"Integer: NA ** 0 = {(na_int ** 0)[0]}")
print(f"Integer: NA * 0 = {(na_int * 0)[0]}")
```

Output:
```
NA ** 0 = 1.0
NA * 0 = <NA>
Integer: NA ** 0 = 1
Integer: NA * 0 = <NA>
```

## Why This Is A Bug

This behavior is inconsistent and violates the principle of uniform NA propagation in pandas:

1. **Inconsistency**: `NA ** 0 = 1` but `NA * 0 = NA`. Both are indeterminate forms where mathematical identities conflict with NA propagation rules.

2. **Violates user expectations**: All other operations with identity elements propagate NA consistently:
   - `NA + 0 = NA` ✓
   - `NA - 0 = NA` ✓
   - `NA * 1 = NA` ✓
   - `NA / 1 = NA` ✓
   - `NA ** 1 = NA` ✓
   - `NA ** 0 = 1` ✗ (inconsistent)

3. **Asymmetric behavior**: `NA ** 0 = 1` but `0 ** NA = NA`, showing the operation is not handled uniformly.

## Fix

The fix should make power operations consistent with other operations by propagating NA. In `pandas/core/arrays/masked.py`, the `__pow__` method should check for NA values in the base before applying the mathematical identity.

```diff
--- a/pandas/core/arrays/masked.py
+++ b/pandas/core/arrays/masked.py
@@ -somewhere in __pow__ or _arith_method
-    result = self._data ** other
+    if np.isscalar(other) and other == 0:
+        # x**0 = 1, but NA**0 should be NA for consistency
+        result = np.where(self._mask, 1, self._data ** other)
+        # Apply mask to result
+    else:
+        result = self._data ** other
```

Note: The exact implementation would need to handle both scalar and array `other` values, and integrate with the existing mask propagation logic in `_arith_method`.