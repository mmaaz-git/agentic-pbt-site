# Bug Report: pandas.core.internals.api.maybe_infer_ndim - Allows Invalid ndim Values

**Target**: `pandas.core.internals.api.maybe_infer_ndim`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `maybe_infer_ndim` function can return ndim values outside the valid range of [1, 2] for pandas Blocks. When `ndim=None` and `values` is a numpy array with `ndim ∉ {1, 2}`, the function returns `values.ndim` without validation, violating the pandas internals contract that all Blocks must be 1D or 2D.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas._libs.internals import BlockPlacement
from pandas.core.internals.api import maybe_infer_ndim

@given(st.integers(min_value=0, max_value=5))
def test_maybe_infer_ndim_returns_only_1_or_2(ndim):
    """Property: maybe_infer_ndim should only return 1 or 2 for numpy arrays"""
    shape = tuple([2] * max(ndim, 1)) if ndim > 0 else ()
    if ndim == 0:
        arr = np.array(5)
    else:
        arr = np.arange(np.prod(shape)).reshape(shape)

    placement = BlockPlacement(slice(0, 1))
    result = maybe_infer_ndim(arr, placement, ndim=None)

    assert result in [1, 2], f"Expected 1 or 2, got {result} for array with ndim={arr.ndim}"
```

**Failing input**: `ndim=0` (scalar array) or `ndim≥3` (3D+ arrays)

## Reproducing the Bug

```python
import numpy as np
from pandas._libs.internals import BlockPlacement
from pandas.core.internals.api import maybe_infer_ndim, make_block

placement = BlockPlacement(slice(0, 1))

# Case 1: 0D array
scalar = np.array(5)
result = maybe_infer_ndim(scalar, placement, ndim=None)
print(f"0D array: maybe_infer_ndim returned {result}, expected 1 or 2")

# Case 2: 3D array
array_3d = np.array([[[1, 2], [3, 4]]])
result_3d = maybe_infer_ndim(array_3d, placement, ndim=None)
print(f"3D array: maybe_infer_ndim returned {result_3d}, expected 1 or 2")

# Downstream impact: Block creation succeeds but BlockManager creation fails
block_3d = make_block(array_3d, placement, ndim=None)
print(f"Block created with ndim={block_3d.ndim}")

from pandas.core.internals.managers import BlockManager
from pandas import Index
try:
    mgr = BlockManager((block_3d,), [Index([0]), Index([0, 1])])
except AssertionError as e:
    print(f"BlockManager creation failed: {e}")
```

## Why This Is A Bug

The pandas Block architecture only supports 1D and 2D blocks, as evidenced by:

1. The BlockManager constructor asserts that block dimensions must match the number of axes (always 2 for DataFrames)
2. Various operations in `pandas.core.internals` assume blocks are 1D or 2D
3. The `maybe_infer_ndim` function is called by `make_block`, which is a "pseudo-public API for downstream libraries" - it should validate inputs to prevent invalid block creation

By allowing `ndim ∉ {1, 2}`, the function creates blocks that will fail later with confusing error messages, making debugging difficult for downstream library authors.

## Fix

Add validation in `maybe_infer_ndim` to ensure the inferred ndim is always 1 or 2:

```diff
--- a/pandas/core/internals/api.py
+++ b/pandas/core/internals/api.py
@@ -98,11 +98,17 @@ def maybe_infer_ndim(values, placement: BlockPlacement, ndim: int | None) -> in
     """
     If `ndim` is not provided, infer it from placement and values.
     """
     if ndim is None:
         # GH#38134 Block constructor now assumes ndim is not None
         if not isinstance(values.dtype, np.dtype):
             if len(placement) != 1:
                 ndim = 1
             else:
                 ndim = 2
         else:
             ndim = values.ndim
+            # Blocks must be 1D or 2D
+            if ndim not in (1, 2):
+                raise ValueError(
+                    f"Blocks must be 1D or 2D, got values with ndim={ndim}"
+                )
     return ndim
```