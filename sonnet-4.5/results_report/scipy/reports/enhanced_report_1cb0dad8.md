# Bug Report: scipy.special.boxcox1p Inconsistent Inverse Transformation for Small Lambda Values

**Target**: `scipy.special.boxcox1p` and `scipy.special.inv_boxcox1p`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `boxcox1p` and `inv_boxcox1p` use different thresholds to determine when lambda is "close enough to zero", causing the documented inverse relationship to fail with ~31% error for lambda values smaller than approximately 1e-150.

## Property-Based Test

```python
import math
from hypothesis import given, strategies as st, settings
from scipy.special import boxcox1p, inv_boxcox1p


@settings(max_examples=1000)
@given(
    y=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    lmbda=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
def test_boxcox1p_round_trip_boxcox_first(y, lmbda):
    x = inv_boxcox1p(y, lmbda)
    if not math.isfinite(x) or x <= -1:
        return
    y_recovered = boxcox1p(x, lmbda)
    assert math.isfinite(y_recovered), f"boxcox1p returned non-finite value: {y_recovered}"
    assert math.isclose(y_recovered, y, rel_tol=1e-9, abs_tol=1e-9), \
        f"Round-trip failed: y={y}, lmbda={lmbda}, x={x}, y_recovered={y_recovered}"

if __name__ == "__main__":
    test_boxcox1p_round_trip_boxcox_first()
```

<details>

<summary>
**Failing input**: `y=1.0, lmbda=9.261058084022965e-181`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 21, in <module>
    test_boxcox1p_round_trip_boxcox_first()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 7, in test_boxcox1p_round_trip_boxcox_first
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 17, in test_boxcox1p_round_trip_boxcox_first
    assert math.isclose(y_recovered, y, rel_tol=1e-9, abs_tol=1e-9), \
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Round-trip failed: y=1.0, lmbda=9.261058084022965e-181, x=1.0, y_recovered=0.6931471805599453
Falsifying example: test_boxcox1p_round_trip_boxcox_first(
    y=1.0,
    lmbda=9.261058084022965e-181,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/17/hypo.py:18
```
</details>

## Reproducing the Bug

```python
from scipy.special import boxcox1p, inv_boxcox1p

y = 1.0
lmbda = 5.808166112732823e-234

x = inv_boxcox1p(y, lmbda)
y_recovered = boxcox1p(x, lmbda)

print(f"Input: y={y}, lambda={lmbda}")
print(f"inv_boxcox1p(y, lambda) = {x}")
print(f"boxcox1p(x, lambda) = {y_recovered}")
print(f"Expected: {y}")
print(f"Actual: {y_recovered}")
print(f"Error: {abs(y_recovered - y)}")
```

<details>

<summary>
Round-trip transformation fails with 31% error
</summary>
```
Input: y=1.0, lambda=5.808166112732823e-234
inv_boxcox1p(y, lambda) = 1.0
boxcox1p(x, lambda) = 0.6931471805599453
Expected: 1.0
Actual: 0.6931471805599453
Error: 0.3068528194400547
```
</details>

## Why This Is A Bug

This violates the documented inverse relationship between `boxcox1p` and `inv_boxcox1p`. The documentation for `inv_boxcox1p` explicitly states it computes "the inverse of the Box-Cox transformation" and should find x such that the Box-Cox equations are satisfied.

The bug manifests when lambda is extremely small (< ~1e-150). In these cases:
- `inv_boxcox1p(1.0, lambda)` returns 1.0, apparently using some threshold to treat lambda as non-zero
- `boxcox1p(1.0, lambda)` returns 0.6931471805599453 (which equals log(2)), using the lambda=0 special case formula: log(1+x) = log(1+1) = log(2)

This inconsistency means the two functions are using different thresholds to decide when to apply the lambda=0 special case. The mathematical definition states:
- When λ ≠ 0: y = ((1+x)^λ - 1) / λ
- When λ = 0: y = log(1+x)

However, the implementation appears to use different epsilon thresholds in each function to determine "lambda ≈ 0", creating a range where they disagree on which formula to use.

## Relevant Context

The threshold appears to be around lambda = 1e-150. Testing shows:
- For lambda >= 1e-150: Functions work correctly (round-trip succeeds with near-zero error)
- For lambda < 1e-155: Functions become inconsistent (round-trip fails with ~31% error)

The returned incorrect value 0.6931471805599453 is exactly log(2), confirming that `boxcox1p` is using the lambda=0 case while `inv_boxcox1p` is not.

Documentation links:
- [SciPy boxcox1p documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox1p.html)
- [SciPy inv_boxcox1p documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.inv_boxcox1p.html)

## Proposed Fix

Since `boxcox1p` and `inv_boxcox1p` are implemented as ufuncs in C/Cython, the fix requires modifying the C source code to use consistent thresholds. Both functions should use the same epsilon value when checking if lambda should be treated as zero. A high-level approach would be:

1. Define a consistent threshold constant (e.g., `BOXCOX_LAMBDA_EPSILON = 1e-150`)
2. In both functions, use `fabs(lambda) < BOXCOX_LAMBDA_EPSILON` to determine when to use the lambda=0 special case
3. Ensure both functions apply the same logic consistently

This would ensure the inverse relationship holds for all valid inputs as documented.