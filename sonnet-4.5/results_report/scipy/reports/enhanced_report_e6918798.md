# Bug Report: scipy.special.inv_boxcox1p Returns Input Unchanged for Extremely Small Lambda Values

**Target**: `scipy.special.inv_boxcox1p`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The function `scipy.special.inv_boxcox1p` incorrectly returns the input value `y` unchanged when lambda values are smaller than approximately 1e-200, instead of computing the correct inverse transformation `exp(y) - 1`, violating the mathematical round-trip property with `boxcox1p`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import scipy.special
import numpy as np
import math

@given(
    x=st.floats(min_value=-0.99, max_value=1e6, allow_nan=False, allow_infinity=False),
    lmbda=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=1000)
def test_boxcox1p_inv_boxcox1p_roundtrip(x, lmbda):
    y = scipy.special.boxcox1p(x, lmbda)
    assume(not np.isnan(y) and not np.isinf(y))
    result = scipy.special.inv_boxcox1p(y, lmbda)
    assert math.isclose(result, x, rel_tol=1e-9, abs_tol=1e-12)

# Run the test
if __name__ == "__main__":
    test_boxcox1p_inv_boxcox1p_roundtrip()
```

<details>

<summary>
**Failing input**: `x=1.0, lmbda=3.658266661562511e-162`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 19, in <module>
    test_boxcox1p_inv_boxcox1p_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 7, in test_boxcox1p_inv_boxcox1p_roundtrip
    x=st.floats(min_value=-0.99, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 15, in test_boxcox1p_inv_boxcox1p_roundtrip
    assert math.isclose(result, x, rel_tol=1e-9, abs_tol=1e-12)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_boxcox1p_inv_boxcox1p_roundtrip(
    x=1.0,
    lmbda=3.658266661562511e-162,
)
```
</details>

## Reproducing the Bug

```python
import scipy.special
import numpy as np

# Test case demonstrating the bug
x = 1.0
lmbda = 1e-264

# Apply boxcox1p transformation
y = scipy.special.boxcox1p(x, lmbda)
print(f"boxcox1p({x}, {lmbda}) = {y}")

# Apply inverse transformation
result = scipy.special.inv_boxcox1p(y, lmbda)
print(f"inv_boxcox1p({y}, {lmbda}) = {result}")
print(f"Expected: {x}")
print(f"Error: {abs(result - x)}")

# Additional test with exactly lambda = 0 to show it works correctly
print("\n--- Testing with lambda = 0 (should work correctly) ---")
y_zero = scipy.special.boxcox1p(x, 0.0)
result_zero = scipy.special.inv_boxcox1p(y_zero, 0.0)
print(f"boxcox1p({x}, 0.0) = {y_zero}")
print(f"inv_boxcox1p({y_zero}, 0.0) = {result_zero}")
print(f"Expected: {x}")
print(f"Error: {abs(result_zero - x)}")

# Test with various small lambda values to find threshold
print("\n--- Testing various small lambda values ---")
for exp in [-100, -200, -250, -260, -264, -300]:
    test_lmbda = 10.0 ** exp
    y_test = scipy.special.boxcox1p(x, test_lmbda)
    result_test = scipy.special.inv_boxcox1p(y_test, test_lmbda)
    error = abs(result_test - x)
    status = "OK" if error < 1e-10 else "FAIL"
    print(f"lambda=1e{exp}: error={error:.15e} [{status}]")
```

<details>

<summary>
Output showing the error
</summary>
```
boxcox1p(1.0, 1e-264) = 0.6931471805599453
inv_boxcox1p(0.6931471805599453, 1e-264) = 0.6931471805599453
Expected: 1.0
Error: 0.3068528194400547

--- Testing with lambda = 0 (should work correctly) ---
boxcox1p(1.0, 0.0) = 0.6931471805599453
inv_boxcox1p(0.6931471805599453, 0.0) = 1.0
Expected: 1.0
Error: 0.0

--- Testing various small lambda values ---
lambda=1e-100: error=2.220446049250313e-16 [OK]
lambda=1e-200: error=3.068528194400547e-01 [FAIL]
lambda=1e-250: error=3.068528194400547e-01 [FAIL]
lambda=1e-260: error=3.068528194400547e-01 [FAIL]
lambda=1e-264: error=3.068528194400547e-01 [FAIL]
lambda=1e-300: error=3.068528194400547e-01 [FAIL]
```
</details>

## Why This Is A Bug

The Box-Cox transformation and its inverse are defined to satisfy the round-trip property: `inv_boxcox1p(boxcox1p(x, λ), λ) = x` for all valid values of `x` and `λ`.

According to the documentation and mathematical definition:
- For `boxcox1p(x, λ)`:
  - When λ ≠ 0: `((1+x)^λ - 1) / λ`
  - When λ = 0: `log(1+x)` (as the limit λ→0)
- For `inv_boxcox1p(y, λ)`:
  - When λ ≠ 0: `(y·λ + 1)^(1/λ) - 1`
  - When λ = 0: `exp(y) - 1` (as the limit λ→0)

The bug manifests when λ is extremely small (< ~1e-200) but non-zero:

1. `boxcox1p` correctly recognizes that for very small λ, the transformation approaches `log(1+x)`. For x=1.0, it returns ln(2) ≈ 0.6931471805599453.

2. However, `inv_boxcox1p` has a flawed threshold detection. When λ < ~1e-200, instead of computing the limiting form `exp(y) - 1`, it incorrectly returns `y` unchanged.

3. This creates a mathematical inconsistency:
   - Input: x = 1.0, λ = 1e-264
   - Forward: y = boxcox1p(1.0, 1e-264) = 0.693147... (correctly uses log(1+x) limiting form)
   - Inverse: inv_boxcox1p(0.693147..., 1e-264) = 0.693147... (WRONG: returns y instead of exp(y)-1)
   - Expected: exp(0.693147...) - 1 = 2.0 - 1 = 1.0

The function works correctly when λ = 0 exactly (returns exp(y) - 1 = 1.0), and when λ > ~1e-200, but fails in the intermediate range where λ is positive but extremely small.

## Relevant Context

The implementation is in a compiled C extension (likely in `scipy/special/_boxcox.c` or similar), as indicated by the Cython bindings in `_ufuncs.pyx`. The bug appears to be in the threshold logic that determines when to use the limiting case formula.

Key observations:
- The threshold appears to be around 1e-200 (works at 1e-100, fails at 1e-200 and smaller)
- When the small-λ branch is triggered, it returns `y` directly instead of computing `exp(y) - 1`
- The λ=0 case is handled correctly with a separate code path
- This affects any lambda value below the threshold, not just specific values

The impact is significant for scientific computing applications that may use optimization algorithms or numerical methods that explore very small lambda values, potentially leading to silent errors in calculations.

## Proposed Fix

The bug is in the implementation logic for handling small lambda values. When the function detects λ is below the threshold for numerical stability, it should compute the limiting form `exp(y) - 1`, not return `y` directly. Here's the conceptual fix:

```diff
// Pseudo-code showing the likely fix needed in the C implementation
double inv_boxcox1p(double y, double lambda) {
    // ... existing code ...

    if (fabs(lambda) < THRESHOLD) {  // e.g., THRESHOLD = 1e-200
-       return y;  // WRONG: returns input unchanged
+       return expm1(y);  // CORRECT: compute exp(y) - 1 using expm1 for numerical stability
    }

    // ... rest of implementation for normal lambda values ...
}
```

The fix should use `expm1(y)` (which computes `exp(y) - 1` with better numerical precision for small values) instead of returning `y` directly when lambda is below the threshold.