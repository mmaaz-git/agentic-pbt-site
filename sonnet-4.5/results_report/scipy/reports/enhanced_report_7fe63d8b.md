# Bug Report: scipy.stats.bartlett Returns NaN P-value for Identical Sample Variances

**Target**: `scipy.stats.bartlett`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.stats.bartlett` function returns NaN for the p-value when all input samples have identical variances, instead of returning the mathematically correct value of 1.0.

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

<details>

<summary>
**Failing input**: `samples=[[0.0, 12581.0, 53.5, 93.25], [0.0, 12581.0, 53.5, 93.25], [0.0, 12581.0, 53.5, 93.25]]`
</summary>
```
Testing specific failing case...
Bartlett statistic: 0.0
Bartlett p-value: nan
Assertion Error as expected: Bartlett p-value nan outside [0, 1]

Running full Hypothesis test suite...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 46, in <module>
    test_bartlett_pvalue_bounds()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 9, in test_bartlett_pvalue_bounds
    samples=st.lists(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 25, in test_bartlett_pvalue_bounds
    assert 0 <= p <= 1, f"Bartlett p-value {p} outside [0, 1]"
           ^^^^^^^^^^^
AssertionError: Bartlett p-value nan outside [0, 1]
Falsifying example: test_bartlett_pvalue_bounds(
    samples=[[0.0, 12581.0, 53.5, 93.25],
     [0.0, 12581.0, 53.5, 93.25],
     [0.0, 12581.0, 53.5, 93.25]],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.stats

# Test case: three identical samples with equal variances
samples = [
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0]),
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 64.0])
]

# Display variance of each sample to show they are equal
for i, sample in enumerate(samples):
    print(f"Sample {i+1} variance: {np.var(sample, ddof=1)}")

print("\nRunning Bartlett test...")
statistic, p = scipy.stats.bartlett(*samples)
print(f"Bartlett statistic: {statistic}")
print(f"Bartlett p-value: {p}")
print(f"P-value is NaN: {np.isnan(p)}")

# Verify what chi-squared survival function returns at 0
from scipy.stats import chi2
print(f"\nExpected p-value (chi2.sf(0, df=2)): {chi2.sf(0, df=2)}")

# Also try with scipy.special directly
from scipy import special
print(f"special.chdtrc(2, 0): {special.chdtrc(2, 0)}")
```

<details>

<summary>
Output: Bartlett test returns NaN p-value for identical variances
</summary>
```
Sample 1 variance: 674.4
Sample 2 variance: 674.4
Sample 3 variance: 674.4

Running Bartlett test...
Bartlett statistic: 0.0
Bartlett p-value: nan
P-value is NaN: True

Expected p-value (chi2.sf(0, df=2)): 1.0
special.chdtrc(2, 0): 1.0
```
</details>

## Why This Is A Bug

The Bartlett test tests the null hypothesis that all input samples have equal variances. When this null hypothesis is perfectly satisfied (all samples have identical variances), the test statistic should be 0 and the p-value should be 1.0.

The bug occurs due to a numerical precision issue combined with incorrect ordering of operations in the code:

1. **Numerical Precision**: When all variances are identical, the numerator calculation `(Ntot - k) * log(spsq) - sum((Ni - 1)*log(ssq))` should theoretically be 0, but due to floating-point arithmetic, it becomes a tiny negative number (e.g., -1.3e-14).

2. **Division**: This tiny negative numerator divided by the denominator produces a tiny negative T statistic.

3. **Chi-squared Distribution**: The `scipy.special.chdtrc` function (chi-squared survival function) returns NaN when given a negative input, even if it's just numerical noise.

4. **Misplaced Clipping**: The code does include `T = xp.clip(T, min=0., max=xp.inf)` to handle this issue, but this clipping happens AFTER the p-value calculation (line 2956), not before (line 2954).

The mathematical expectation is clear:
- For a chi-squared distribution with any degrees of freedom, the survival function at 0 equals 1.0
- P-values must be valid probabilities in the range [0, 1]
- NaN is never a valid p-value

## Relevant Context

The issue is located in `/scipy/stats/_morestats.py` in the `bartlett` function. The specific problematic code section (lines 2951-2960):

```python
T = numer / denom  # Can be slightly negative due to numerical precision

chi2 = _SimpleChi2(xp.asarray(k-1))
pvalue = _get_pvalue(T, chi2, alternative='greater', symmetric=False, xp=xp)  # Returns NaN if T < 0

T = xp.clip(T, min=0., max=xp.inf)  # This clipping comes too late!
```

The `special.chdtrc` function used internally by `_SimpleChi2.sf()` returns NaN for negative inputs, which is mathematically correct behavior. The bug is in the bartlett function not handling numerical precision issues before calling the chi-squared distribution.

## Proposed Fix

The fix is simple: move the clipping operation before the p-value calculation to ensure the statistic is never negative when passed to the chi-squared distribution.

```diff
--- a/scipy/stats/_morestats.py
+++ b/scipy/stats/_morestats.py
@@ -2949,11 +2949,12 @@ def bartlett(*samples, axis=0):
     denom = (1 + 1/(3*(k - 1))
              * ((xp.sum(1/(Ni - 1), axis=0)) - 1/(Ntot - k)))
     T = numer / denom
+    # Clip to avoid negative values due to numerical precision
+    T = xp.clip(T, min=0., max=xp.inf)

     chi2 = _SimpleChi2(xp.asarray(k-1))
     pvalue = _get_pvalue(T, chi2, alternative='greater', symmetric=False, xp=xp)

-    T = xp.clip(T, min=0., max=xp.inf)
     T = T[()] if T.ndim == 0 else T
     pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
```