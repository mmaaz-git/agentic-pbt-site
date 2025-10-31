# Bug Report: pandas.core.array_algos.masked_accumulations Input Mutation

**Target**: `pandas.core.array_algos.masked_accumulations`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

All accumulation functions in `masked_accumulations` (`cumsum`, `cumprod`, `cummin`, `cummax`) mutate the input `values` array in-place, violating the fundamental expectation that input arrays should remain unchanged unless explicitly documented as in-place operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.array_algos import masked_accumulations

@given(
    values=st.lists(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False), min_size=1, max_size=100),
    mask_indices=st.data()
)
def test_cumsum_does_not_mutate_input(values, mask_indices):
    values_arr = np.array(values, dtype=np.float64)
    original_copy = values_arr.copy()

    mask = np.zeros(len(values), dtype=bool)
    indices = mask_indices.draw(st.lists(st.integers(min_value=0, max_value=len(values)-1), max_size=len(values)//2))
    for idx in indices:
        mask[idx] = True

    result, result_mask = masked_accumulations.cumsum(values_arr, mask, skipna=True)

    assert np.array_equal(values_arr, original_copy), \
        f"Input array was mutated! Original: {original_copy}, After: {values_arr}"
```

**Failing input**: Any array with at least one masked value

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos import masked_accumulations

arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
mask = np.array([False, True, False, True, False], dtype=bool)

print("Before cumsum:", arr)
result, result_mask = masked_accumulations.cumsum(arr, mask, skipna=True)
print("After cumsum:", arr)
print("Result:", result)
```

**Expected output**:
```
Before cumsum: [1. 2. 3. 4. 5.]
After cumsum: [1. 2. 3. 4. 5.]  # Input unchanged
Result: [1. 1. 4. 4. 9.]  # Cumsum with masked values treated as 0
```

**Actual output**:
```
Before cumsum: [1. 2. 3. 4. 5.]
After cumsum: [1. 0. 3. 0. 5.]  # Input mutated!
Result: [1. 1. 4. 4. 9.]
```

## Why This Is A Bug

The bug is on **line 68** of `masked_accumulations.py`:

```python
def _cum_func(
    func: Callable,
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
):
    # ... determine fill_value ...

    values[mask] = fill_value  # BUG: Mutates input array!

    if not skipna:
        mask = np.maximum.accumulate(mask)

    values = func(values)  # Returns new array
    return values, mask
```

**Problem**: The function directly modifies the input `values` array on line 68 before calling the accumulation function. While `func(values)` returns a new array (line 73), the original input array passed by the caller has already been mutated.

**Impact**:
1. **Unexpected side effects**: Callers don't expect their arrays to be modified
2. **Data corruption**: Original data is permanently altered
3. **Difficult debugging**: Mutation happens silently without warning
4. **Violates pandas conventions**: Most pandas functions don't mutate inputs unless explicitly documented

While the docstring on line 29 states "We will modify values in place", this is an implementation detail that should not affect the caller's input array.

## Fix

Create a copy of the input array before mutation:

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

Alternatively, if the mutation is intentional for performance, the docstring and function signature should be updated to make this explicit:

```python
def _cum_func(
    func: Callable,
    values: np.ndarray,  # Will be modified in-place
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
):
    """
    Accumulations for 1D masked array.

    WARNING: This function modifies the input values array in-place!
    Callers should pass a copy if they need to preserve the original.

    ...
    """
```