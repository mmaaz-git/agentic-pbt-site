# Bug Report: scipy.stats.percentileofscore Returns Values > 100

**Target**: `scipy.stats.percentileofscore`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `percentileofscore` function can return values slightly greater than 100 due to floating point arithmetic errors, violating the documented constraint that percentiles should be in the range [0, 100].

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
import scipy.stats as stats

@given(
    npst.arrays(
        dtype=float,
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=5, max_side=100),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    ),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=300)
def test_percentileofscore_bounds(arr, score):
    percentile = stats.percentileofscore(arr, score)
    assert 0 <= percentile <= 100, \
        f"Percentile should be in [0, 100], got {percentile}"
```

**Failing input**: `arr=array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])`, `score=0.0`

## Reproducing the Bug

```python
import numpy as np
import scipy.stats as stats

arr = np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
score = 0.0

percentile = stats.percentileofscore(arr, score)

print(f"Percentile: {percentile}")
print(f"Percentile > 100: {percentile > 100}")
```

Output:
```
Percentile: 100.00000000000001
Percentile > 100: True
```

The bug affects all `kind` parameters ('rank', 'weak', 'strict', 'mean').

## Why This Is A Bug

According to the documentation, `percentileofscore` computes "the percentile rank of a score relative to a list of scores." Percentile ranks are defined as values in the range [0, 100], representing percentages. The function should ensure that the returned value never exceeds 100 (or falls below 0) even when floating point arithmetic introduces small errors.

This violates the API contract that percentiles are in [0, 100] and can cause issues in downstream code that assumes this invariant (e.g., validation checks, plotting libraries that expect percentiles in [0, 100]).

## Fix

The fix should clamp the output to ensure it stays within [0, 100]:

```diff
--- a/scipy/stats/_stats_py.py
+++ b/scipy/stats/_stats_py.py
@@ -2000,7 +2000,7 @@ def percentileofscore(a, score, kind='rank', nan_policy='propagate'):
     else:
         pct = (left + right + (1 if kind == 'strict' else 0)) * 100.0 / n

-    return pct
+    return np.clip(pct, 0.0, 100.0)
```

Alternatively, the calculation could be adjusted to avoid the floating point error in the first place, but clamping is simpler and more robust.