# Bug Report: scipy.stats.poisson.ppf Returns Infinity for CDF=1.0

**Target**: `scipy.stats.poisson.ppf`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ppf` (percent point function) method of scipy.stats' Poisson distribution returns `inf` when the CDF equals exactly 1.0 due to floating-point precision. This violates the expected property that `ppf(cdf(k))` should return a finite value close to `k`.

## Property-Based Test

```python
import scipy.stats as stats
from hypothesis import given, strategies as st, settings

@given(st.floats(min_value=0.1, max_value=50, allow_nan=False, allow_infinity=False),
       st.integers(min_value=0, max_value=100))
@settings(max_examples=300)
def test_poisson_ppf_cdf_property(mu, k):
    cdf_k = stats.poisson.cdf(k, mu)
    ppf_result = stats.poisson.ppf(cdf_k, mu)

    # ppf should return a finite value
    assert ppf_result != float('inf'), f"poisson.ppf(cdf({k})) = {ppf_result}, should be finite"
```

**Failing input**: `mu=1.0, k=18`

## Reproducing the Bug

```python
import scipy.stats as stats

mu, k = 1.0, 18

cdf_k = stats.poisson.cdf(k, mu)
print(f"cdf({k}) = {cdf_k}")
print(f"cdf({k}) == 1.0: {cdf_k == 1.0}")

ppf_result = stats.poisson.ppf(cdf_k, mu)
print(f"ppf(cdf({k})) = {ppf_result}")

assert ppf_result != float('inf')
```

Output:
```
cdf(18) = 1.0
cdf(18) == 1.0: True
ppf(cdf(18)) = inf
AssertionError
```

## Why This Is A Bug

For discrete distributions, the percent point function (PPF) is defined as:

```
ppf(q) = min{k: cdf(k) >= q}
```

In this case:
- For a Poisson distribution with `mu=1.0`, when `k=18`, the CDF rounds to exactly `1.0` in floating-point arithmetic
- The theoretical CDF is `1 - 7.48e-18`, which is indistinguishable from 1.0 in float64
- `ppf(1.0)` should return `18` (the minimum k where `cdf(k) >= 1.0`), but instead returns `inf`

This is problematic because:
1. It breaks the round-trip property: users expect that `ppf(cdf(k))` returns a value close to `k`
2. It's inconsistent with how finite-support distributions handle this edge case
3. It makes ppf unreliable for values of q very close to 1.0

The Poisson distribution has infinite support (k can be arbitrarily large), but when the CDF numerically equals 1.0, the ppf should return the smallest k that achieved this, not infinity.

## Fix

The ppf implementation should not special-case `q == 1.0` to return infinity. Instead, it should use a search algorithm to find the minimum k where `cdf(k) >= q`, which will naturally return a finite value when the CDF has numerically converged to 1.0.