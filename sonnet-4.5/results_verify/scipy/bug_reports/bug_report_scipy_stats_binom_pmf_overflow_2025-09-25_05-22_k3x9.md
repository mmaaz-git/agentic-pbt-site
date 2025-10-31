# Bug Report: scipy.stats.binom.pmf OverflowError with Extremely Small Probabilities

**Target**: `scipy.stats.binom.pmf`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.stats.binom.pmf` raises an `OverflowError` when computing probabilities with extremely small values of p (near machine epsilon ~1e-308), even though the computation is mathematically well-defined and should return a valid result.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import scipy.stats

@given(
    st.integers(min_value=1, max_value=100),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=300)
def test_binom_edge_probabilities(n, p):
    pmf_0 = scipy.stats.binom.pmf(0, n, p)
    pmf_n = scipy.stats.binom.pmf(n, n, p)

    assert pmf_0 >= 0
    assert pmf_n >= 0

    if p == 0:
        assert pmf_0 == 1.0
        assert pmf_n == 0.0
    elif p == 1:
        assert pmf_0 == 0.0
        assert pmf_n == 1.0
```

**Failing input**: `n=3, p=1.1125369292536007e-308`

## Reproducing the Bug

```python
import scipy.stats

n = 3
p = 1.1125369292536007e-308

scipy.stats.binom.pmf(0, n, p)
```

**Output:**
```
OverflowError: Error in function ibeta_derivative<d>(%1%,%1%,%1%): Overflow Error
```

**Expected:** Should return approximately `1.0` (when p is extremely small, getting 0 successes is almost certain).

**Additional observations:**
- The bug occurs when `p <= 1e-308` (near minimum representable float)
- It occurs for `n >= 3`, but works fine for `n = 1` or `n = 2`
- For `p >= 1e-300`, the function works correctly and returns ~1.0 as expected

## Why This Is A Bug

1. The binomial PMF is mathematically well-defined for all valid inputs (0 <= p <= 1, n >= 1)
2. When p is extremely small, `pmf(0, n, p) ≈ (1-p)^n ≈ 1`, which is a valid result
3. The function should handle extreme but valid inputs gracefully without raising exceptions
4. This violates the API contract that pmf should work for all valid probability values in [0, 1]

## Fix

The overflow occurs in the underlying `ibeta_derivative` function from the boost library. The fix should add special case handling for extremely small p values before calling the beta derivative:

```diff
diff --git a/scipy/stats/_discrete_distns.py b/scipy/stats/_discrete_distns.py
index 1234567..abcdefg 100644
--- a/scipy/stats/_discrete_distns.py
+++ b/scipy/stats/_discrete_distns.py
@@ -80,6 +80,11 @@ class binom_gen(stats.rv_discrete):
     def _pmf(self, k, n, p):
         # Binomial PMF implementation
+        # Handle extreme p values to avoid overflow in beta derivative
+        if p < 1e-300:
+            # For very small p, pmf(k) ≈ C(n,k) * p^k * (1-p)^(n-k) ≈ (1-p)^(n-k)
+            return np.where(k == 0, 1.0, 0.0)
+
         # Original implementation...
         return _boost.binom_pmf(k, n, p)
```

Alternatively, the underlying boost library call should be wrapped with try-except to gracefully handle overflow errors at extreme parameter values.