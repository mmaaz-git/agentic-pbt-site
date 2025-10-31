# Bug Report: scipy.special.expit/logit Round-Trip Asymmetry

**Target**: `scipy.special.expit` and `scipy.special.logit`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip property `logit(expit(x)) = x`, which is mathematically exact and documented as an inverse relationship, fails catastrophically for large positive x values (≥20) but works perfectly for large negative x values. This asymmetry indicates a numerical stability bug in the implementation.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import scipy.special
import numpy as np

@given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
@settings(max_examples=1000)
def test_expit_logit_round_trip(x):
    result = scipy.special.logit(scipy.special.expit(x))
    assert np.isclose(result, x, rtol=1e-9, atol=1e-9), \
        f"logit(expit({x})) = {result}, expected {x}"
```

**Failing input**: `x=20.0`

## Reproducing the Bug

```python
import numpy as np
import scipy.special

print("For large NEGATIVE x (works perfectly):")
for x in [-10, -20, -30, -40, -50, -100]:
    result = scipy.special.logit(scipy.special.expit(x))
    error = abs(result - x)
    print(f"  logit(expit({x:4})) = {result:8.2f}, error = {error:.2e}")

print("\nFor large POSITIVE x (catastrophic failure):")
for x in [10, 20, 30, 40, 50, 100]:
    result = scipy.special.logit(scipy.special.expit(x))
    if np.isfinite(result):
        error = abs(result - x)
        print(f"  logit(expit({x:4})) = {result:8.2f}, error = {error:.2e}")
    else:
        print(f"  logit(expit({x:4})) = {result:>8}, error = inf")
```

**Output:**
```
For large NEGATIVE x (works perfectly):
  logit(expit( -10)) =   -10.00, error = 0.00e+00
  logit(expit( -20)) =   -20.00, error = 0.00e+00
  logit(expit( -30)) =   -30.00, error = 0.00e+00
  logit(expit( -40)) =   -40.00, error = 0.00e+00
  logit(expit( -50)) =   -50.00, error = 0.00e+00
  logit(expit(-100)) =  -100.00, error = 0.00e+00

For large POSITIVE x (catastrophic failure):
  logit(expit(  10)) =    10.00, error = 9.70e-13
  logit(expit(  20)) =    20.00, error = 3.59e-08
  logit(expit(  30)) =    30.00, error = 1.02e-03
  logit(expit(  40)) =      inf, error = inf
  logit(expit(  50)) =      inf, error = inf
  logit(expit( 100)) =      inf, error = inf
```

## Why This Is A Bug

1. **Documentation violation**: The `expit` docstring explicitly states "It is the inverse of the logit function," and the `logit` docstring states "`expit` is the inverse of `logit`."

2. **Mathematical property**: Mathematically, `logit(expit(x)) = x` should hold exactly for all real x.

3. **Asymmetric behavior**: The function works perfectly for large negative values but fails for large positive values. This asymmetry indicates a fixable implementation issue, not a fundamental numerical limitation.

4. **Root cause**: When x is large and positive, `expit(x) = 1/(1+exp(-x))` returns a value very close to 1. The subsequent computation of `1-p` in `logit(p) = log(p/(1-p))` loses precision through catastrophic cancellation:

```python
x = 20.0
p = scipy.special.expit(x)

one_minus_p_naive = 1 - p
one_minus_p_accurate = np.exp(-x) / (1 + np.exp(-x))

logit_naive = np.log(p / one_minus_p_naive)
logit_accurate = np.log(p / one_minus_p_accurate)

print(f"Using naive 1-p:     {logit_naive:.10f}, error = {abs(logit_naive - x):.2e}")
print(f"Using accurate 1-p:  {logit_accurate:.10f}, error = {abs(logit_accurate - x):.2e}")
```

**Output:**
```
Using naive 1-p:     19.9999999641, error = 3.59e-08
Using accurate 1-p:  20.0000000000, error = 0.00e+00
```

5. **User impact**: Users relying on the documented inverse relationship will get incorrect results for inputs as small as x=20, which is well within normal usage ranges for logistic functions in machine learning and statistics.

## Fix

The issue stems from catastrophic cancellation when computing `1-p` where `p = expit(x)` is very close to 1. For large positive x:
- `expit(x) = 1/(1+exp(-x)) ≈ 1`
- `1 - expit(x)` should be `exp(-x)/(1+exp(-x))`, but computing it as `1 - p` loses precision

The fix requires modifying the C implementation of these ufuncs to use numerically stable formulas. One approach:

For `logit(p)` when p is close to 1, instead of computing `log(p/(1-p))`, use an alternative formulation that avoids the subtraction. This could involve:

1. Detecting when `p > threshold` (e.g., 0.999) and using a special formula
2. Using log1p where appropriate: `log(p/(1-p)) = -log((1-p)/p) = -log1p((1-p-p)/p)`
3. For the specific case of the round-trip, recognizing that `logit(1/(1+exp(-x)))` should simplify directly to `x` algebraically

Since these are ufuncs implemented in C (in `scipy/special/cephes/` or similar), the fix should be made at that level using standard numerical stability techniques for extreme values.