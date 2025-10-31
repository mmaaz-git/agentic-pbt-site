# Bug Report: xarray.polyval Silently Drops Negative Degree Coefficients

**Target**: `xarray.computation.computation.polyval`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `xarray.polyval` function silently drops polynomial coefficients with negative degree indices during reindexing, producing incorrect mathematical results without any warning or error message.

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

if __name__ == "__main__":
    # Run the test to find a failing example
    test_polyval_preserves_all_coefficients()
```

<details>

<summary>
**Failing input**: `coord_data=array([1.])`, `min_degree=-1`, `max_degree=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 30, in <module>
    test_polyval_preserves_all_coefficients()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 7, in test_polyval_preserves_all_coefficients
    coord_data=arrays(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 26, in test_polyval_preserves_all_coefficients
    assert np.allclose(result.values, expected, rtol=1e-10)
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_polyval_preserves_all_coefficients(
    # The test always failed when commented parts were varied together.
    coord_data=array([1.]),  # or any other generated value
    min_degree=-1,  # or any other generated value
    max_degree=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import xarray as xr

# Create test data with negative degree coefficient
coord = xr.DataArray([2.0], dims=("x",))
coeffs = xr.DataArray(
    [100.0, 1.0, 2.0],
    dims=("degree",),
    coords={"degree": [-1, 0, 1]}
)

# Evaluate the polynomial
result = xr.polyval(coord, coeffs)

# Calculate expected result manually
# For x = 2.0:
# Degree -1: 100 * (2.0)^(-1) = 100 * 0.5 = 50.0
# Degree 0:  1 * (2.0)^0 = 1 * 1 = 1.0
# Degree 1:  2 * (2.0)^1 = 2 * 2 = 4.0
# Expected total: 50.0 + 1.0 + 4.0 = 55.0

print(f"Input coordinate: {coord.values[0]}")
print(f"Coefficients: {coeffs.values} with degrees {list(coeffs.degree.values)}")
print(f"Result from xr.polyval: {result.values[0]}")
print(f"Expected result: 100*(2^-1) + 1*(2^0) + 2*(2^1) = 50.0 + 1.0 + 4.0 = 55.0")
print(f"Actual result: {result.values[0]}")
print(f"\nError: The coefficient for degree -1 (value 100.0) was silently dropped!")
print(f"The function only computed: 1*(2^0) + 2*(2^1) = 1.0 + 4.0 = 5.0")
```

<details>

<summary>
Output showing incorrect result: 5.0 instead of expected 55.0
</summary>
```
Input coordinate: 2.0
Coefficients: [100.   1.   2.] with degrees [np.int64(-1), np.int64(0), np.int64(1)]
Result from xr.polyval: 5.0
Expected result: 100*(2^-1) + 1*(2^0) + 2*(2^1) = 50.0 + 1.0 + 4.0 = 55.0
Actual result: 5.0

Error: The coefficient for degree -1 (value 100.0) was silently dropped!
The function only computed: 1*(2^0) + 2*(2^1) = 1.0 + 4.0 = 5.0
```
</details>

## Why This Is A Bug

This behavior violates expected mathematical correctness and API consistency:

1. **Silent data loss**: The function accepts DataArrays with negative degree coordinates but silently ignores these coefficients during computation, providing no warning or error to users.

2. **Mathematical incorrectness**: Laurent polynomials (polynomials with negative degree terms) are a well-established mathematical concept. The function produces mathematically incorrect results for these valid polynomial representations.

3. **API inconsistency**: The function validates that degree coordinates are integers (lines 830-833) but doesn't validate non-negativity. If negative degrees aren't supported, they should be explicitly rejected with an error.

4. **Documentation gap**: Neither the xarray documentation nor the referenced numpy.polynomial.polynomial.polyval documentation explicitly states that negative degrees are unsupported, leading users to reasonably expect they would work.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/computation/computation.py` at lines 834-837:

- Line 834: `max_deg = coeffs[degree_dim].max().item()` - Only captures the maximum degree
- Lines 835-837: `coeffs.reindex({degree_dim: np.arange(max_deg + 1)}, ...)` - Reindexes to range [0, 1, ..., max_deg], dropping any negative degree coefficients

The function uses Horner's method for polynomial evaluation (lines 841-845), which could theoretically handle negative degrees if the reindexing step preserved them.

Documentation: https://docs.xarray.dev/en/stable/generated/xarray.polyval.html

## Proposed Fix

```diff
--- a/xarray/computation/computation.py
+++ b/xarray/computation/computation.py
@@ -831,8 +831,18 @@ def polyval(
         raise ValueError(
             f"Dimension `{degree_dim}` should be of integer dtype. Received {coeffs[degree_dim].dtype} instead."
         )
+
+    min_deg = coeffs[degree_dim].min().item()
     max_deg = coeffs[degree_dim].max().item()
+
+    if min_deg < 0:
+        raise ValueError(
+            f"Polynomial coefficients must have non-negative degrees. "
+            f"Found minimum degree: {min_deg}. "
+            f"Laurent polynomials (with negative degrees) are not supported."
+        )
+
     coeffs = coeffs.reindex(
         {degree_dim: np.arange(max_deg + 1)}, fill_value=0, copy=False
     )
+
     coord = _ensure_numeric(coord)
```