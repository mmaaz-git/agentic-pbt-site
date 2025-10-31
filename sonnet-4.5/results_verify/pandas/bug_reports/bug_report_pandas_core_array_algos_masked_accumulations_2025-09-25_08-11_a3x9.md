# Bug Report: pandas.core.array_algos.masked_accumulations Input Array Mutation

**Target**: `pandas.core.array_algos.masked_accumulations`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The accumulation functions (`cumsum`, `cumprod`, `cummin`, `cummax`) in `masked_accumulations.py` mutate the input `values` array, violating the fundamental expectation that function arguments should remain unchanged unless explicitly intended as in-place operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.array_algos.masked_accumulations import cumsum

@given(
    values=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    mask_indices=st.sets(st.integers(min_value=0, max_value=49), max_size=50)
)
def test_cumsum_does_not_mutate_input(values, mask_indices):
    arr = np.array(values[:min(len(values), 50)], dtype=np.int64)
    mask = np.array([i in mask_indices for i in range(len(arr))], dtype=bool)

    original_arr = arr.copy()

    result_values, result_mask = cumsum(arr, mask, skipna=True)

    assert np.array_equal(arr, original_arr), \
        f"cumsum mutated input array! Before: {original_arr}, After: {arr}"
```

**Failing input**: Any array with masked values

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos.masked_accumulations import cumsum

arr = np.array([10, 20, 30, 40, 50], dtype=np.int64)
mask = np.array([False, True, False, False, True], dtype=bool)

print("Before:", arr)

result_values, result_mask = cumsum(arr, mask, skipna=True)

print("After:", arr)
print("Result:", result_values)
```

Output:
```
Before: [10 20 30 40 50]
After: [10  0 30 40  0]
Result: [10 10 40 80 80]
```

The input array `arr` is modified: values at masked positions are changed to the fill value (0 for cumsum).

## Why This Is A Bug

**Source code analysis of `_cum_func` in masked_accumulations.py (lines 19-74):**

1. **Line 68**: `values[mask] = fill_value`
   - This modifies the input array in-place by setting masked positions to fill_value
   - The fill_value is 0 for cumsum, 1 for cumprod, dtype.min for cummax, dtype.max for cummin

2. **Line 73**: `values = func(values)`
   - This calls the numpy function (e.g., `np.cumsum`) which returns a NEW array
   - The local variable `values` is reassigned to point to this new array
   - **However**, the original input array was already mutated on line 68

3. **Line 74**: `return values, mask`
   - Returns the new array, but the caller's original array is already modified

The function's docstring says "We will modify values in place" but this appears to describe the internal algorithm, not the API contract. Other pandas functions that modify inputs have `_inplace` in their names or explicit `inplace` parameters.

**Impact:**
- Callers who reuse the input array will see unexpected modifications
- This violates the principle of least surprise
- Can lead to subtle bugs in downstream code that doesn't expect mutation

## Fix

The function should make a copy before modifying:

```diff
def _cum_func(
    func: Callable,
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
):
+   # Create a copy to avoid mutating the input
+   values = values.copy()
+
    dtype_info: np.iinfo | np.finfo
    if values.dtype.kind == "f":
        dtype_info = np.finfo(values.dtype.type)
    elif values.dtype.kind in "iu":
        dtype_info = np.iinfo(values.dtype.type)
    elif values.dtype.kind == "b":
        dtype_info = np.iinfo(np.uint8)
    else:
        raise NotImplementedError(
            f"No masked accumulation defined for dtype {values.dtype.type}"
        )
    try:
        fill_value = {
            np.cumprod: 1,
            np.maximum.accumulate: dtype_info.min,
            np.cumsum: 0,
            np.minimum.accumulate: dtype_info.max,
        }[func]
    except KeyError:
        raise NotImplementedError(
            f"No accumulation for {func} implemented on BaseMaskedArray"
        )

    values[mask] = fill_value

    if not skipna:
        mask = np.maximum.accumulate(mask)

    values = func(values)
    return values, mask
```

Alternatively, if in-place modification is intentional for performance, the function should:
1. Be renamed to `_cum_func_inplace` or similar
2. Have explicit documentation stating it mutates inputs
3. Return the modified input array, not a new one (for consistency)