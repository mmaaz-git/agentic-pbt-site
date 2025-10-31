# Bug Report: xarray.polyval Negative Degree Coefficients Silently Dropped

**Target**: `xarray.computation.computation.polyval`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `polyval` function silently drops polynomial coefficients with negative degree indices during reindexing, leading to incorrect results without any warning or error.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import xarray as xr

@given(
    coord_data=arrays(
        dtype=np.float64,
        shape=st.integers(1, 10),
        elements=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
    ),
    min_degree=st.integers(-5, -1),
    max_degree=st.integers(0, 3),
)
def test_polyval_preserves_all_coefficients(coord_data, min_degree, max_degree):
    degrees = list(range(min_degree, max_degree + 1))
    coeffs_data = np.random.uniform(-10, 10, len(degrees))

    coord = xr.DataArray(coord_data, dims=("x",))
    coeffs = xr.DataArray(coeffs_data, dims=("degree",), coords={"degree": degrees})

    result = xr.polyval(coord, coeffs)

    expected = sum(c * coord_data**d for c, d in zip(coeffs_data, degrees))

    assert np.allclose(result.values, expected, rtol=1e-10)
```

**Failing input**: `coord_data=[2.0]`, `coeffs with degrees=[-1, 0, 1]` and `values=[100.0, 1.0, 2.0]`

## Reproducing the Bug

```python
import numpy as np
import xarray as xr

coord = xr.DataArray([2.0], dims=("x",))
coeffs = xr.DataArray(
    [100.0, 1.0, 2.0],
    dims=("degree",),
    coords={"degree": [-1, 0, 1]}
)

result = xr.polyval(coord, coeffs)

print(f"Result: {result.values[0]}")
print(f"Expected: 100/2 + 1 + 2*2 = 55")
print(f"Actual: {result.values[0]}")
```

Output:
```
Result: 5.0
Expected: 100/2 + 1 + 2*2 = 55
Actual: 5.0
```

The coefficient for degree -1 (value 100.0) was silently dropped!

## Why This Is A Bug

The `polyval` function accepts coefficients with explicit degree labels but silently ignores negative degrees:

1. **Line 834**: `max_deg = coeffs[degree_dim].max().item()` gets the maximum degree
2. **Line 835-837**: Reindexes to `np.arange(max_deg + 1)`, which only includes `[0, 1, ..., max_deg]`
3. Any coefficients with degree < 0 are silently dropped during reindexing

This violates user expectations because:
- The function accepts integer degree coordinates without restriction
- Users explicitly label their coefficients with degree values
- No error or warning is raised when coefficients are dropped
- The result is mathematically incorrect

## Fix

```diff
--- a/xarray/computation/computation.py
+++ b/xarray/computation/computation.py
@@ -831,6 +831,11 @@ def polyval(
         raise ValueError(
             f"Dimension `{degree_dim}` should be of integer dtype. Received {coeffs[degree_dim].dtype} instead."
         )
+    min_deg = coeffs[degree_dim].min().item()
+    if min_deg < 0:
+        raise ValueError(
+            f"Polynomial coefficients must have non-negative degrees. Found minimum degree: {min_deg}"
+        )
     max_deg = coeffs[degree_dim].max().item()
     coeffs = coeffs.reindex(
         {degree_dim: np.arange(max_deg + 1)}, fill_value=0, copy=False
```