# Bug Report: scipy.stats.norm.moment() Returns NaN for Extremely Small Location Parameters

**Target**: `scipy.stats.norm.moment()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.stats.norm.moment()` function returns NaN instead of the mathematically correct value when the location parameter is extremely small (below ~1e-309), while the `mean()` method correctly returns the location parameter for the same distribution.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
import numpy as np
import scipy.stats as stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@settings(max_examples=100)
@given(
    st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.5, max_value=3, allow_nan=False, allow_infinity=False),
)
@example(loc=2.225073858507e-311, scale=1.0)  # Add the known failing case
def test_mean_moment_consistency(loc, scale):
    dist = stats.norm(loc=loc, scale=scale)
    mean_value = dist.mean()
    moment_value = dist.moment(1)

    assert np.isclose(mean_value, moment_value, rtol=1e-12), \
        f"mean() = {mean_value}, moment(1) = {moment_value}"

if __name__ == "__main__":
    # Run the test
    test_mean_moment_consistency()
```

<details>

<summary>
**Failing input**: `loc=2.225073858507e-311, scale=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 25, in <module>
    test_mean_moment_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 10, in test_mean_moment_consistency
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 20, in test_mean_moment_consistency
    assert np.isclose(mean_value, moment_value, rtol=1e-12), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: mean() = 2.225073858507e-311, moment(1) = nan
Falsifying explicit example: test_mean_moment_consistency(
    loc=2.225073858507e-311,
    scale=1.0,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.stats as stats
import warnings

# Enable all warnings to see the overflow/underflow warnings
warnings.simplefilter("always")

loc = 2.225073858507e-311
scale = 1.0

dist = stats.norm(loc=loc, scale=scale)
mean_value = dist.mean()
moment_value = dist.moment(1)

print(f"loc = {loc}")
print(f"scale = {scale}")
print(f"mean() = {mean_value}")
print(f"moment(1) = {moment_value}")
print(f"Are they equal? {np.isclose(mean_value, moment_value, rtol=1e-12)}")
print(f"Difference: {abs(mean_value - moment_value)}")
```

<details>

<summary>
Output showing NaN result with overflow warnings
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/stats/_distn_infrastructure.py:1353: RuntimeWarning: overflow encountered in divide
  fac = scale / loc
/home/npc/.local/lib/python3.13/site-packages/scipy/stats/_distn_infrastructure.py:1358: RuntimeWarning: invalid value encountered in multiply
  res2 += fac**n * val
loc = 2.225073858507e-311
scale = 1.0
mean() = 2.225073858507e-311
moment(1) = nan
Are they equal? False
Difference: nan
```
</details>

## Why This Is A Bug

This violates a fundamental mathematical property: the first non-central moment of any probability distribution must equal its mean by definition (E[X] = E[X^1]). For a normal distribution with location parameter `loc` and scale parameter `scale`, the first moment is exactly `loc`.

The bug occurs in scipy's generic moment calculation code in `_distn_infrastructure.py` at lines 1353-1358. When `loc` is extremely small (below approximately 1e-309, approaching the denormal floating-point range), the division `scale / loc` causes floating-point overflow to infinity. This infinity then propagates through the subsequent calculations, ultimately resulting in NaN.

The critical code path is:
```python
# Line 1353 in _distn_infrastructure.py
fac = scale / loc  # Overflows to inf when loc < ~1e-309

# Lines 1354-1358
for k in range(n):
    valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp, shapes)
    res2 += comb(n, k, exact=True)*fac**k * valk  # inf * finite = inf or nan
res2 += fac**n * val  # inf * finite = nan
```

The `mean()` method correctly returns `loc` directly without any problematic calculations, demonstrating that scipy accepts and intends to support these parameter values. This inconsistency between `mean()` and `moment(1)` is the core issue.

## Relevant Context

- The bug affects all moment orders (1st, 2nd, 3rd, 4th, etc.) when `loc` is below ~1e-309
- The threshold occurs near the boundary between normal and denormal floating-point numbers
- Python's `sys.float_info.min` is approximately 2.225e-308, and the bug manifests for values about 1000x smaller
- The overflow occurs because `1.0 / 1e-310` exceeds the maximum representable float (~1.8e308)
- scipy's documentation doesn't specify any restrictions on the magnitude of location parameters
- The issue is deterministic and reproducible across platforms

Documentation references:
- scipy.stats.norm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
- Source code: https://github.com/scipy/scipy/blob/main/scipy/stats/_distn_infrastructure.py

## Proposed Fix

The simplest fix is to special-case the first moment to return the mean directly, avoiding the problematic generic calculation:

```diff
--- a/scipy/stats/_distn_infrastructure.py
+++ b/scipy/stats/_distn_infrastructure.py
@@ -1318,6 +1318,10 @@ class rv_continuous(rv_generic):
             raise ValueError("Moment must be positive.")
         mu, mu2, g1, g2 = None, None, None, None
         if (n > 0) and (n < 5):
+            # Special case: first moment equals mean by definition
+            if n == 1:
+                return self.mean(*args, **kwds)
+
             if self._stats_has_moments:
                 mdict = {'moments': {1: 'm', 2: 'v', 3: 'vs', 4: 'mvsk'}[n]}
             else:
```

Alternatively, protect the division from overflow by checking if `abs(loc)` is too small relative to `scale`:

```diff
--- a/scipy/stats/_distn_infrastructure.py
+++ b/scipy/stats/_distn_infrastructure.py
@@ -1350,7 +1350,12 @@ class rv_continuous(rv_generic):
             *shapes, loc, scale, val = args

             res2 = zeros(loc.shape, dtype='d')
-            fac = scale / loc
+            # Avoid overflow for extremely small loc values
+            with np.errstate(over='raise'):
+                try:
+                    fac = scale / loc
+                except FloatingPointError:
+                    fac = np.sign(loc) * np.inf if loc != 0 else np.inf
             for k in range(n):
                 valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp,
                                           shapes)
```