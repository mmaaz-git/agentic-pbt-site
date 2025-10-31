# Bug Report: scipy.cluster.vq.whiten Float Precision Bug

**Target**: `scipy.cluster.vq.whiten`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `whiten` function incorrectly handles constant-valued columns when floating point precision causes `np.std()` to return a very small non-zero value instead of exactly 0. This causes the function to divide by a tiny number, producing astronomically large incorrect values instead of preserving the original constant values as documented.

## Property-Based Test

```python
import numpy as np
from hypothesis import given
from hypothesis.extra import numpy as npst
from hypothesis import strategies as st, settings, assume
from scipy.cluster.vq import whiten


@given(npst.arrays(dtype=np.float64,
                   shape=npst.array_shapes(min_dims=2, max_dims=2,
                                          min_side=2, max_side=100),
                   elements=st.floats(min_value=-1e6, max_value=1e6,
                                     allow_nan=False, allow_infinity=False)))
@settings(max_examples=500)
def test_whiten_unit_variance(obs):
    std_devs = np.std(obs, axis=0)
    assume(np.all(std_devs > 1e-10))

    result = whiten(obs)

    result_std = np.std(result, axis=0)

    for i, s in enumerate(result_std):
        assert abs(s - 1.0) < 1e-6, f"Column {i} has std {s}, expected 1.0"
```

**Failing input**: Array with all constant values `[[93206.82233024, 93206.82233024], ...]`

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import whiten

obs = np.array([[93206.82233024, 93206.82233024]] * 40)

print(f"Input: all values = {obs[0, 0]}")
print(f"Std: {np.std(obs, axis=0)}")
print(f"Std == 0: {np.std(obs, axis=0) == 0}")

result = whiten(obs)

print(f"Output: {result[0, 0]}")
print(f"Expected: {obs[0, 0]} (unchanged)")
print(f"Actual change: {obs[0, 0]} → {result[0, 0]}")
print(f"Multiplication factor: {result[0, 0] / obs[0, 0]}")
```

Output:
```
Input: all values = 93206.82233024
Std: [5.82076609e-11 5.82076609e-11]
Std == 0: [False False]
Output: 1601281014689853.2
Expected: 93206.82233024 (unchanged)
Actual change: 93206.82233024 → 1.60128101e+15
Multiplication factor: 17179869184.0
```

## Why This Is A Bug

According to the function's documentation (lines 143-146 in `vq.py`):

> "Some columns have standard deviation zero. The values of these columns will not change."

However, when an array has constant values, `np.std()` may return a very small non-zero value (e.g., 5.82e-11) due to floating point precision errors in the calculation. The code at line 141 uses exact equality:

```python
zero_std_mask = std_dev == 0
```

This comparison fails when `std_dev` is 5.82e-11 instead of exactly 0, causing the function to divide by this tiny number on line 147:

```python
return obs / std_dev
```

This produces astronomically large incorrect values (e.g., 93206.82 / 5.82e-11 = 1.60e+15) instead of preserving the constant values.

## Fix

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -138,7 +138,8 @@ def whiten(obs, check_finite=None):
         check_finite = not is_lazy_array(obs)
     obs = _asarray(obs, check_finite=check_finite, xp=xp)
     std_dev = xp.std(obs, axis=0)
-    zero_std_mask = std_dev == 0
+    # Use tolerance to handle floating point precision issues
+    zero_std_mask = std_dev < 1e-10 * xp.max(xp.abs(obs), axis=0) + 1e-14
     std_dev = xpx.at(std_dev, zero_std_mask).set(1.0)
     if check_finite and xp.any(zero_std_mask):
         warnings.warn("Some columns have standard deviation zero. "
```

The fix uses a tolerance-based comparison that accounts for floating point precision. The threshold `1e-10 * max(abs(obs)) + 1e-14` is relative to the data magnitude with an absolute floor, which correctly identifies columns with effectively zero variance while avoiding false positives.