# Bug Report: scipy.cluster.vq.whiten Floating-Point Precision Bug with Constant Columns

**Target**: `scipy.cluster.vq.whiten`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `whiten` function fails to handle columns with identical values due to floating-point precision issues, producing values 15 orders of magnitude too large instead of leaving the column unchanged as promised by its warning message.

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
        f"Column with zero std should remain unchanged. Got {whitened[:, zero_col]} instead of {constant_value}"

# Run the test
test_whiten_zero_std_columns_unchanged()
```

<details>

<summary>
**Failing input**: `n_obs=7, n_features=1`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:16: RuntimeWarning: Some columns have standard deviation zero. The values of these columns will not change.
  whitened = whiten(obs)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 22, in <module>
    test_whiten_zero_std_columns_unchanged()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 6, in test_whiten_zero_std_columns_unchanged
    st.integers(min_value=1, max_value=10))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 18, in test_whiten_zero_std_columns_unchanged
    assert np.allclose(whitened[:, zero_col], constant_value), \
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Column with zero std should remain unchanged. Got [6.44159549e+15 6.44159549e+15 6.44159549e+15 6.44159549e+15
 6.44159549e+15 6.44159549e+15 6.44159549e+15] instead of 5.721286105539075
Falsifying example: test_whiten_zero_std_columns_unchanged(
    n_obs=7,
    n_features=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/39/hypo.py:19
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1016
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1021
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import whiten

# Create an array with 7 identical values
obs = np.array([[5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075],
                [5.721286105539075]])

# Check the standard deviation
std_val = np.std(obs, axis=0)[0]
print(f"std_val: {std_val}")
print(f"std_val == 0: {std_val == 0}")

# Apply whiten function
whitened = whiten(obs)
print(f"Original value: {obs[0, 0]}")
print(f"Whitened value: {whitened[0, 0]}")

# Verify all values are identical
print(f"All values identical: {np.all(obs == obs[0, 0])}")

# Mathematical verification
print(f"Division result: {obs[0, 0] / std_val}")
```

<details>

<summary>
Output showing extreme value inflation
</summary>
```
std_val: 8.881784197001252e-16
std_val == 0: False
Original value: 5.721286105539075
Whitened value: 6441595493246444.0
All values identical: True
Division result: 6441595493246444.0
```
</details>

## Why This Is A Bug

The `whiten` function contains a warning message at line 144-145 that explicitly promises: "Some columns have standard deviation zero. **The values of these columns will not change.**" However, when a column contains all identical floating-point values, numpy's standard deviation calculation returns a near-zero value (8.88e-16) due to floating-point rounding errors rather than exactly zero.

The code uses exact equality comparison (`std_dev == 0`) at line 141 to detect zero standard deviation columns. This fails to catch these near-zero values, causing the function to divide by the tiny standard deviation and produce astronomically incorrect results - values that are 15 orders of magnitude larger than the original (6.44e15 instead of 5.72).

This violates the behavioral contract established by the function's own warning message and breaks reasonable user expectations that columns with mathematically zero variance should remain unchanged.

## Relevant Context

The bug occurs in `/scipy/cluster/vq.py` at lines 140-147. The current implementation:
1. Computes standard deviation using `xp.std(obs, axis=0)`
2. Checks for exact zero with `zero_std_mask = std_dev == 0`
3. Sets std_dev to 1.0 for zero-std columns to avoid division
4. Issues a warning about unchanged columns
5. Returns `obs / std_dev`

The issue arises because floating-point arithmetic can produce values like 8.88e-16 for the standard deviation of identical values, which is mathematically zero but computationally non-zero. This is a well-known issue in numerical computing that requires tolerance-based comparisons.

Columns with constant values are common in real datasets (e.g., features that haven't varied in a sample, categorical encodings, or data subsets), making this a practical concern for users of the library.

## Proposed Fix

```diff
--- a/scipy/cluster/vq.py
+++ b/scipy/cluster/vq.py
@@ -138,7 +138,7 @@ def whiten(obs, check_finite=None):
         check_finite = not is_lazy_array(obs)
     obs = _asarray(obs, check_finite=check_finite, xp=xp)
     std_dev = xp.std(obs, axis=0)
-    zero_std_mask = std_dev == 0
+    zero_std_mask = xp.abs(std_dev) < 1e-10
     std_dev = xpx.at(std_dev, zero_std_mask).set(1.0)
     if check_finite and xp.any(zero_std_mask):
         warnings.warn("Some columns have standard deviation zero. "
```