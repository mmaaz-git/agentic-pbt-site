# Bug Report: scipy.special.boxcox/inv_boxcox Subnormal Lambda Round-Trip Failure

**Target**: `scipy.special.boxcox` and `scipy.special.inv_boxcox`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `boxcox` and `inv_boxcox` functions fail to maintain their documented inverse relationship when lambda is a subnormal float (e.g., 5e-324), causing silent calculation errors in mathematical transformations.

## Property-Based Test

```python
import math
from hypothesis import given, strategies as st, settings
from scipy import special

@given(
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=1000)
def test_boxcox_inv_boxcox_round_trip(x, lmbda):
    y = special.boxcox(x, lmbda)
    result = special.inv_boxcox(y, lmbda)
    assert math.isclose(result, x, rel_tol=1e-9, abs_tol=1e-9)

if __name__ == "__main__":
    test_boxcox_inv_boxcox_round_trip()
```

<details>

<summary>
**Failing input**: `x=2.0, lmbda=5e-324`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 16, in <module>
    test_boxcox_inv_boxcox_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 6, in test_boxcox_inv_boxcox_round_trip
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 13, in test_boxcox_inv_boxcox_round_trip
    assert math.isclose(result, x, rel_tol=1e-9, abs_tol=1e-9)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_boxcox_inv_boxcox_round_trip(
    x=2.0,
    lmbda=5e-324,
)
```
</details>

## Reproducing the Bug

```python
from scipy import special
import numpy as np

x = 0.5
lmbda = 5e-324

y = special.boxcox(x, lmbda)
print(f"boxcox(0.5, 5e-324) = {y}")
print(f"Expected log(0.5) = {np.log(0.5)}")

result = special.inv_boxcox(y, lmbda)
print(f"inv_boxcox({y}, 5e-324) = {result}")
print(f"Expected: 0.5")
print(f"Actual: {result}")

print(f"\nWith exact zero:")
y_zero = special.boxcox(x, 0.0)
result_zero = special.inv_boxcox(y_zero, 0.0)
print(f"inv_boxcox(boxcox(0.5, 0.0), 0.0) = {result_zero}")
```

<details>

<summary>
Output demonstrating incorrect inverse computation
</summary>
```
boxcox(0.5, 5e-324) = -0.6931471805599453
Expected log(0.5) = -0.6931471805599453
inv_boxcox(-0.6931471805599453, 5e-324) = 0.36787944117144233
Expected: 0.5
Actual: 0.36787944117144233

With exact zero:
inv_boxcox(boxcox(0.5, 0.0), 0.0) = 0.5
```
</details>

## Why This Is A Bug

The Box-Cox transformation and its inverse are fundamental mathematical operations in scipy.special. According to the documentation, `inv_boxcox` should find `x` such that `y = boxcox(x, lmbda)`, establishing them as inverse functions.

When lambda is a subnormal float (5e-324):
1. `boxcox(x, 5e-324)` correctly treats the subnormal lambda as effectively zero, returning `log(x)` as per the lambda=0 special case
2. `inv_boxcox(y, 5e-324)` fails to use the same threshold, producing incorrect results that don't match `exp(y)`
3. The round-trip property `inv_boxcox(boxcox(x, lmbda), lmbda) == x` is violated

Testing shows `inv_boxcox` with subnormal lambda produces erratic results:
- For x=0.5: returns 0.3679 (1/e) instead of 0.5
- For x=2.0: returns 2.7183 (e) instead of 2.0
- For x=4.0: returns 2.7183 (e) instead of 4.0

This breaks mathematical correctness in scientific computing applications relying on these transformations.

## Relevant Context

The Box-Cox transformation is defined as:
- `y = (x^lambda - 1) / lambda` when lambda ≠ 0
- `y = log(x)` when lambda = 0

Subnormal floats are values so small they lose precision but aren't exactly zero. The smallest positive normal float64 is approximately 2.2e-308, while subnormal floats range from about 5e-324 to 2.2e-308.

The bug manifests because:
- `boxcox` appears to have a threshold check treating very small lambda as 0
- `inv_boxcox` either lacks this threshold or uses a different one
- This inconsistency causes the functions to use different formulas for the same subnormal lambda value

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox.html

## Proposed Fix

The fix requires ensuring both `boxcox` and `inv_boxcox` use identical logic for detecting when lambda should be treated as zero. A high-level approach:

1. Define a consistent epsilon threshold (e.g., `BOXCOX_LAMBDA_TOL = 1e-10`)
2. Both functions should check: `if abs(lambda) < BOXCOX_LAMBDA_TOL: use_zero_formula`
3. Alternatively, explicitly handle subnormal floats: check if lambda is subnormal using floating-point classification

Without access to the C implementation, the exact patch cannot be provided, but the fix should ensure both functions apply the lambda=0 special case using the same criteria, maintaining the mathematical inverse relationship for all valid inputs including subnormal values.