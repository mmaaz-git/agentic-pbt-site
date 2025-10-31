# Bug Report: scipy.stats.bartlett Returns NaN P-value

**Target**: `scipy.stats.bartlett`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `scipy.stats.bartlett` function returns NaN for the p-value when all input samples have identical variances, instead of returning 1.0 as expected from the chi-squared distribution.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.stats
from hypothesis import given, strategies as st, assume, settings

@given(
    samples=st.lists(
        st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=3, max_size=50),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=300)
def test_bartlett_pvalue_bounds(samples):
    samples = [np.array(s) for s in samples]
    STD_THRESHOLD = 1e-6

    for s in samples:
        assume(len(s) >= 3)
        assume(np.std(s) > STD_THRESHOLD)

    statistic, p = scipy.stats.bartlett(*samples)
    assert 0 <= p <= 1, f"Bartlett p-value {p} outside [0, 1]"
```

**Failing input**:
```python
samples = [
    [0.0, 0.0, 0.0, 1.0, 1.0, 64.0],
    [0.0, 0.0, 0.0, 1.0, 1.0, 64.0],
    [0.0, 0.0, 0.0, 1.0, 1.0, 64.0]
]
```

## Reproducing the Bug

```python
import numpy as np
import scipy.stats

samples = [
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0])
]

statistic, p = scipy.stats.bartlett(*samples)
print(f"Statistic: {statistic}")
print(f"P-value: {p}")
```

Output:
```
Statistic: 0.0
P-value: nan
```

## Why This Is A Bug

The Bartlett test tests the null hypothesis that all input samples have equal variances. The test statistic follows a chi-squared distribution with k-1 degrees of freedom (where k is the number of samples).

When all samples have identical variances (as in the failing example where all three samples are identical), the Bartlett statistic is 0.0, which is the expected value under perfect equality of variances.

For a chi-squared distribution, the survival function at 0 should return 1.0:
```python
from scipy.stats import chi2
chi2.sf(0, df=2)  # Returns: 1.0
```

However, `bartlett` returns NaN instead of 1.0. This indicates a division by zero or other numerical issue in the p-value computation when the statistic is exactly 0.

## Fix

The issue likely occurs in the computation of the p-value from the Bartlett statistic. The fix should handle the special case when the statistic is 0 (or very close to 0):

```diff
--- a/scipy/stats/_morestats.py
+++ b/scipy/stats/_morestats.py
@@ -XXXX,X +XXXX,X @@ def bartlett(...):
     ...
     # Compute test statistic
     T = ... # Bartlett statistic computation

     # Compute p-value from chi-squared distribution
+    # Handle special case when T = 0 (perfect equality of variances)
+    if T == 0.0 or T < 1e-14:
+        p = 1.0
+    else:
+        p = distributions.chi2.sf(T, k - 1)
-    p = distributions.chi2.sf(T, k - 1)
     return T, p
```

Alternatively, the chi-squared survival function itself could be made more robust to handle the case when the statistic is exactly 0.