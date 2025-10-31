# Bug Report: scipy.cluster.vq.whiten Floating-Point Precision Bug

**Target**: `scipy.cluster.vq.whiten`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `whiten` function fails to handle columns with near-zero standard deviation due to floating-point precision issues, producing extremely large values instead of leaving the column unchanged as documented.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.cluster.vq import whiten

@given(st.integers(min_value=2, max_value=20),
       st.integers(min_value=1, max_value=10))
@settings(max_examples=200)
def test_whiten_zero_std_columns_unchanged(n_obs, n_features):
    rng = np.random.default_rng(42)
    obs = rng.standard_normal((n_obs, n_features))

    zero_col = rng.integers(0, n_features)
    constant_value = rng.uniform(-10, 10)
    obs[:, zero_col] = constant_value

    whitened = whiten(obs)

    assert np.allclose(whitened[:, zero_col], constant_value), \
        "Column with zero std should remain unchanged"
```

**Failing input**: `n_obs=7, n_features=1`

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import whiten

obs = np.array([[5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075]])

std_val = np.std(obs, axis=0)[0]
print(f"std_val: {std_val}")
print(f"std_val == 0: {std_val == 0}")

whitened = whiten(obs)
print(f"Original value: {obs[0, 0]}")
print(f"Whitened value: {whitened[0, 0]}")
```

Output:
```
std_val: 8.881784197001252e-16
std_val == 0: False
Whitened value: 6441595493246444.0
```

## Why This Is A Bug

The documentation and warning message explicitly state: "The values of these columns will not change" for columns with zero standard deviation. However, due to floating-point precision, a column with all identical values may have a computed standard deviation that is very close to zero (e.g., 8.88e-16) but not exactly zero.

The code at line 141 in `vq.py` uses exact equality comparison:
```python
zero_std_mask = std_dev == 0
```

This fails to catch near-zero values, leading to division by extremely small numbers and producing astronomically large results (6.44e15 instead of 5.72).

## Fix

Replace the exact equality check with a tolerance-based comparison:

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -138,7 +138,7 @@ def whiten(obs, check_finite=None):
         check_finite = not is_lazy_array(obs)
     obs = _asarray(obs, check_finite=check_finite, xp=xp)
     std_dev = xp.std(obs, axis=0)
-    zero_std_mask = std_dev == 0
+    zero_std_mask = std_dev < 1e-10
     std_dev = xpx.at(std_dev, zero_std_mask).set(1.0)
     if check_finite and xp.any(zero_std_mask):
         warnings.warn("Some columns have standard deviation zero. "
```

An alternative fix would use `np.isclose` or a multiple of machine epsilon for the threshold.