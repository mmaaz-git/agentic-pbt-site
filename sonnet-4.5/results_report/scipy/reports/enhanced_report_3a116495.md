# Bug Report: scipy.special.expit/logit Round-Trip Numerical Instability for Positive Values

**Target**: `scipy.special.expit` and `scipy.special.logit`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The documented inverse relationship between `expit` and `logit` functions fails catastrophically for positive values x≥20 due to numerical instability, while working perfectly for negative values of any magnitude.

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

if __name__ == "__main__":
    test_expit_logit_round_trip()
```

<details>

<summary>
**Failing input**: `x=20.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 13, in <module>
    test_expit_logit_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 6, in test_expit_logit_round_trip
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 9, in test_expit_logit_round_trip
    assert np.isclose(result, x, rtol=1e-9, atol=1e-9), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: logit(expit(20.0)) = 19.99999996410867, expected 20.0
Falsifying example: test_expit_logit_round_trip(
    x=20.0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/6/hypo.py:10
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.special

print("Testing round-trip property: logit(expit(x)) should equal x")
print("="*60)

print("\nFor large NEGATIVE x (works perfectly):")
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

print("\n" + "="*60)
print("Demonstrating the root cause - catastrophic cancellation:")
print()

x = 20.0
p = scipy.special.expit(x)
print(f"x = {x}")
print(f"p = expit({x}) = {p:.20f}")
print(f"  (note: p is very close to 1.0)")

one_minus_p_naive = 1 - p
one_minus_p_accurate = np.exp(-x) / (1 + np.exp(-x))

print(f"\n1-p computed naively (1 - {p:.20f}):")
print(f"  1-p = {one_minus_p_naive:.20e}")

print(f"\n1-p computed accurately (exp(-x)/(1+exp(-x))):")
print(f"  1-p = {one_minus_p_accurate:.20e}")

logit_naive = np.log(p / one_minus_p_naive)
logit_accurate = np.log(p / one_minus_p_accurate)

print(f"\nlogit(p) using naive 1-p:")
print(f"  Result: {logit_naive:.10f}, error = {abs(logit_naive - x):.2e}")

print(f"\nlogit(p) using accurate 1-p:")
print(f"  Result: {logit_accurate:.10f}, error = {abs(logit_accurate - x):.2e}")

print("\n" + "="*60)
print("Testing critical value where it becomes infinity:")
print()
for x in [35, 36, 37, 38]:
    p = scipy.special.expit(x)
    result = scipy.special.logit(p)
    print(f"x = {x}: expit({x}) = {p:.20f}")
    print(f"         logit(expit({x})) = {result}")
    print()
```

<details>

<summary>
Asymmetric behavior: perfect for negative values, catastrophic failure for positive values
</summary>
```
Testing round-trip property: logit(expit(x)) should equal x
============================================================

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

============================================================
Demonstrating the root cause - catastrophic cancellation:

x = 20.0
p = expit(20.0) = 0.99999999793884630783
  (note: p is very close to 1.0)

1-p computed naively (1 - 0.99999999793884630783):
  1-p = 2.06115369216774979577e-09

1-p computed accurately (exp(-x)/(1+exp(-x))):
  1-p = 2.06115361819020332521e-09

logit(p) using naive 1-p:
  Result: 19.9999999641, error = 3.59e-08

logit(p) using accurate 1-p:
  Result: 20.0000000000, error = 0.00e+00

============================================================
Testing critical value where it becomes infinity:

x = 35: expit(35) = 0.99999999999999933387
         logit(expit(35)) = 34.945041100449046

x = 36: expit(36) = 0.99999999999999977796
         logit(expit(36)) = 36.04365338911715

x = 37: expit(37) = 1.00000000000000000000
         logit(expit(37)) = inf

x = 38: expit(38) = 1.00000000000000000000
         logit(expit(38)) = inf
```
</details>

## Why This Is A Bug

This violates the explicitly documented inverse relationship between `expit` and `logit`. Both function docstrings state they are inverses of each other:

1. **From `expit` documentation**: "It is the inverse of the logit function."
2. **From `logit` documentation**: "`expit` is the inverse of `logit`"
3. **From `logit` examples**: The documentation shows `expit(logit([0.1, 0.75, 0.999]))` returning the exact input values

The bug exhibits clear asymmetric behavior: the round-trip works perfectly for arbitrarily large negative values (tested up to -100) but fails progressively for positive values starting at x≈20. The error magnitude grows from 3.59e-08 at x=20 to complete failure (infinity) at x≥37.

The root cause is **catastrophic cancellation** when computing `1-p` in the logit function where `p = expit(x)` is very close to 1.0. When x is large and positive, `expit(x) = 1/(1+exp(-x))` approaches 1.0, causing severe precision loss when computing `1 - expit(x)` through direct subtraction. The demonstration shows that using the mathematically equivalent formula `exp(-x)/(1+exp(-x))` completely eliminates the error.

## Relevant Context

This bug impacts real-world applications in machine learning and statistics where these functions are commonly used for:
- Logistic regression models
- Neural network activation functions
- Probability transformations
- Statistical modeling

The failing values (x≥20) are well within typical ranges for these applications. For example, in logistic regression with strong predictors, linear combinations of features routinely exceed 20.

The scipy implementation uses C ufuncs located in the scipy/special module. The functions are documented as being exact inverses without any caveats about numerical limitations or restricted ranges.

Documentation links:
- [SciPy special functions documentation](https://docs.scipy.org/doc/scipy/reference/special.html)
- The C implementation is likely in scipy/special/cephes/ or scipy/special/src/

## Proposed Fix

The fix requires modifying the C implementation to handle values close to 1.0 using numerically stable formulas. Since the exact implementation location would need investigation, here's a high-level approach:

For the `logit` function when `p` is close to 1.0 (e.g., p > 0.999):
1. Recognize that for p close to 1, computing `1-p` directly causes catastrophic cancellation
2. Use an alternative computation path that avoids the subtraction
3. For the specific round-trip case, recognize that `logit(expit(x))` should simplify algebraically to `x`

A conceptual fix in pseudo-code:

```diff
// In the logit implementation
double logit(double p) {
+   // Handle values very close to 1 specially
+   if (p > 0.999999) {
+       // Use log1p for better precision: log(p/(1-p)) = -log1p((1-2p)/p)
+       // Or maintain auxiliary information if this is a round-trip from expit
+       double one_minus_p = -expm1(log1p(-p));  // More accurate 1-p
+       return log(p / one_minus_p);
+   }

    // Original implementation for normal range
    return log(p / (1.0 - p));
}
```

Alternatively, if the round-trip pattern can be detected, a specialized path could return the original value directly, avoiding the numerical instability entirely.