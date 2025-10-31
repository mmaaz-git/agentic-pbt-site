# Bug Report: pandas.core.array_algos.masked_accumulations Input Array Mutation

**Target**: `pandas.core.array_algos.masked_accumulations.cumsum` (and related functions: `cumprod`, `cummin`, `cummax`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The masked accumulation functions (`cumsum`, `cumprod`, `cummin`, `cummax`) unexpectedly modify the input array when mask contains `True` values, despite returning a new array as the result. This violates the principle of least surprise and can lead to subtle bugs in calling code.

## Property-Based Test

```python
import numpy as np
from pandas.core.array_algos.masked_accumulations import cumsum
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst


@given(
    values=npst.arrays(dtype=np.int64, shape=st.integers(5, 20),
                      elements=st.integers(-100, 100))
)
def test_cumsum_should_not_mutate_input(values):
    """cumsum should not modify the input array."""
    original = values.copy()
    mask = np.zeros(len(values), dtype=bool)
    mask[len(values) // 2] = True  # Set at least one mask value to True

    result_values, result_mask = cumsum(values, mask, skipna=True)

    # This assertion fails!
    np.testing.assert_array_equal(
        values, original,
        err_msg="cumsum modified the input array!"
    )
```

**Failing input**: Any array with a mask containing at least one `True` value.

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos.masked_accumulations import cumsum

values = np.array([1, 2, 3, 4, 5])
mask = np.array([False, True, False, True, False])

print("Before cumsum:", values)

result_values, result_mask = cumsum(values, mask, skipna=True)

print("After cumsum:", values)
print("Result:", result_values)
```

**Output:**
```
Before cumsum: [1 2 3 4 5]
After cumsum: [1 0 3 0 5]
Result: [1 1 4 4 9]
```

The input array `values` has been modified: positions where `mask=True` have been set to 0 (the fill value for cumsum).

## Why This Is A Bug

This behavior violates the API contract in several ways:

1. **Unexpected side effects**: Functions that return a result should not modify their inputs unless clearly documented and necessary. While the internal docstring mentions "modify values in place", this is contradicted by the fact that the function returns a new array.

2. **Inconsistent behavior**: When `mask` is all `False`, the input is not modified. When `mask` has any `True` values, the input is modified. This inconsistency is confusing.

3. **Memory safety**: The calling code may reuse the input array after the function call, expecting it to be unchanged. This can lead to subtle bugs where downstream code operates on corrupted data.

4. **Principle of least surprise**: Most NumPy-style functions either:
   - Modify in-place and return the same array (e.g., `array.sort()`)
   - Create a new array and leave the input unchanged (e.g., `np.cumsum()`)

   This function does both, which is unexpected.

## Fix

The fix is to copy the values array before modifying it:

```diff
--- a/pandas/core/array_algos/masked_accumulations.py
+++ b/pandas/core/array_algos/masked_accumulations.py
@@ -65,7 +65,8 @@ def _cum_func(
             f"No accumulation for {func} implemented on BaseMaskedArray"
         )

-    values[mask] = fill_value
+    values = values.copy()
+    values[mask] = fill_value

     if not skipna:
         mask = np.maximum.accumulate(mask)
```

This ensures the input array is never modified, aligning with NumPy conventions and user expectations.