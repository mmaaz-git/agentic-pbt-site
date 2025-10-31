# Bug Report: xarray.corr Returns Correlation Values Outside Valid Range [-1, 1]

**Target**: `xarray.corr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `xarray.corr()` function returns correlation coefficients slightly outside the mathematically valid range of [-1, 1] due to floating-point precision errors, violating the fundamental mathematical constraint of Pearson correlation.

## Property-Based Test

```python
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import xarray as xr

@given(
    shape=st.tuples(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10)),
    data_a=st.data(),
    data_b=st.data(),
)
@settings(max_examples=200)
def test_corr_bounded(shape, data_a, data_b):
    arr_a = data_a.draw(
        arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )
    arr_b = data_b.draw(
        arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )

    da_a = xr.DataArray(arr_a, dims=["x", "y"])
    da_b = xr.DataArray(arr_b, dims=["x", "y"])

    assume(da_a.std().item() > 1e-10)
    assume(da_b.std().item() > 1e-10)

    result = xr.corr(da_a, da_b)

    assert np.all(result.values >= -1.0) and np.all(result.values <= 1.0), f"Correlation {result.values} is outside [-1, 1]"

if __name__ == "__main__":
    test_corr_bounded()
```

<details>

<summary>
**Failing input**: `shape=(3, 3), arr_a=arr_b=[[0., 29., 0.], [0., 0., 0.], [1., 1., 1.]]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 39, in <module>
    test_corr_bounded()
    ~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 7, in test_corr_bounded
    shape=st.tuples(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10)),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 36, in test_corr_bounded
    assert np.all(result.values >= -1.0) and np.all(result.values <= 1.0), f"Correlation {result.values} is outside [-1, 1]"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Correlation 1.0000000000000002 is outside [-1, 1]
Falsifying example: test_corr_bounded(
    shape=(3, 3),
    data_a=data(...),
    data_b=data(...),
)
Draw 1: array([[ 0., 29.,  0.],
       [ 0.,  0.,  0.],
       [ 1.,  1.,  1.]])
Draw 2: array([[ 0., 29.,  0.],
       [ 0.,  0.,  0.],
       [ 1.,  1.,  1.]])
```
</details>

## Reproducing the Bug

```python
import numpy as np
import xarray as xr

# Reproduce the exact bug case
data_a = np.array([[65., 65.,  0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
data_b = np.array([[65., 65.,  0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

da_a = xr.DataArray(data_a, dims=["x", "y"])
da_b = xr.DataArray(data_b, dims=["x", "y"])

# Check that the arrays have nonzero standard deviation
print(f"da_a standard deviation: {da_a.std().item()}")
print(f"da_b standard deviation: {da_b.std().item()}")

# Compute correlation
result = xr.corr(da_a, da_b)

# Display results
print(f"\nCorrelation result: {result.values}")
print(f"Result type: {type(result.values)}")
print(f"Result > 1.0: {result.values > 1.0}")
print(f"Result == 1.0: {result.values == 1.0}")
print(f"Difference from 1.0: {result.values - 1.0}")

# Check if it's in the valid range [-1, 1]
is_valid = -1.0 <= result.values <= 1.0
print(f"\nIs correlation in valid range [-1, 1]? {is_valid}")

# Compare with numpy's corrcoef
numpy_result = np.corrcoef(data_a.flatten(), data_b.flatten())[0, 1]
print(f"\nNumPy's corrcoef result: {numpy_result}")
print(f"NumPy result == 1.0: {numpy_result == 1.0}")
print(f"NumPy result > 1.0: {numpy_result > 1.0}")
```

<details>

<summary>
Output showing correlation value exceeding 1.0
</summary>
```
da_a standard deviation: 21.496729402399797
da_b standard deviation: 21.496729402399797

Correlation result: 1.0000000000000002
Result type: <class 'numpy.ndarray'>
Result > 1.0: True
Result == 1.0: False
Difference from 1.0: 2.220446049250313e-16

Is correlation in valid range [-1, 1]? False

NumPy's corrcoef result: 1.0
NumPy result == 1.0: True
NumPy result > 1.0: False
```
</details>

## Why This Is A Bug

The Pearson correlation coefficient is mathematically defined to be in the range [-1, 1]. Any value outside this range violates this fundamental property. Specifically:

1. **Mathematical invariant violation**: The correlation coefficient must satisfy |ρ| ≤ 1 by definition. The value 1.0000000000000002 exceeds this bound.

2. **Downstream code failures**: Code that depends on correlation being in [-1, 1] will fail. For example:
   - `np.arccos(result.values)` would raise a ValueError: "math domain error"
   - Statistical tests that assume valid correlation bounds would produce incorrect results
   - Visualization code expecting correlations in [-1, 1] may fail or display incorrectly

3. **Inconsistency with other libraries**: NumPy's `corrcoef` and pandas' `Series.corr` both correctly return 1.0 for the same input data, properly handling floating-point precision.

4. **Not an edge case**: This occurs with straightforward data patterns (arrays with repeated values), not just pathological inputs.

## Relevant Context

The bug originates in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/computation/computation.py:312` where the correlation is computed as:
```python
corr = cov / (da_a_std * da_b_std)
```

This direct division can accumulate floating-point errors that push the result slightly outside [-1, 1]. The xarray documentation states that `corr` "Computes the Pearson correlation coefficient" which by mathematical definition must be in [-1, 1].

Other statistical libraries handle this by clamping the final result. For example, NumPy's implementation effectively ensures the result stays within bounds through its computation method.

## Proposed Fix

```diff
--- a/xarray/computation/computation.py
+++ b/xarray/computation/computation.py
@@ -309,5 +309,6 @@ def _cov_corr(
         else:
             da_a_std = da_a.std(dim=dim)
             da_b_std = da_b.std(dim=dim)
         corr = cov / (da_a_std * da_b_std)
-        return cast(T_DataArray, corr)
+        # Clamp to [-1, 1] to handle floating-point precision issues
+        corr_clamped = corr.clip(-1, 1)
+        return cast(T_DataArray, corr_clamped)
```