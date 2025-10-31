# Bug Report: scipy.special.pseudo_huber Returns NaN for Small Delta Values Due to Numerical Overflow

**Target**: `scipy.special.pseudo_huber`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `pseudo_huber` function returns NaN instead of finite values when delta is smaller than approximately 1e-190, violating its mathematical definition due to numerical overflow in the intermediate computation (r/delta)².

## Property-Based Test

```python
import scipy.special as sp
import numpy as np
from hypothesis import given, strategies as st, settings

@settings(max_examples=2000)
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1e-308, max_value=10),
       st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10))
def test_pseudo_huber_returns_finite(delta, r):
    result = sp.pseudo_huber(delta, r)
    assert np.isfinite(result), f"pseudo_huber({delta}, {r}) returned {result}, expected finite value"

if __name__ == "__main__":
    test_pseudo_huber_returns_finite()
```

<details>

<summary>
**Failing input**: `delta=1e-308, r=1.0`
</summary>
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from hypo import test_pseudo_huber_returns_finite; test_pseudo_huber_returns_finite()
                                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 6, in test_pseudo_huber_returns_finite
    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=1e-308, max_value=10),
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 10, in test_pseudo_huber_returns_finite
    assert np.isfinite(result), f"pseudo_huber({delta}, {r}) returned {result}, expected finite value"
           ~~~~~~~~~~~^^^^^^^^
AssertionError: pseudo_huber(1e-308, 1.0) returned nan, expected finite value
Falsifying example: test_pseudo_huber_returns_finite(
    delta=1e-308,
    r=1.0,
)
```
</details>

## Reproducing the Bug

```python
import scipy.special as sp
import numpy as np

# Test cases showing NaN returns for small delta values
print("Testing scipy.special.pseudo_huber with small delta values:")
print("=" * 60)

# Test case from the bug report
delta = 1e-200
r = 1.0
result = sp.pseudo_huber(delta, r)
print(f"sp.pseudo_huber({delta}, {r}) = {result}")

# Additional test cases to show the pattern
test_cases = [
    (1e-100, 1.0),
    (1e-150, 1.0),
    (1e-190, 1.0),
    (1e-200, 1.0),
    (1e-250, 1.0),
    (1e-300, 1.0),
    (2.3581411596114265e-203, 1.0)  # Example from hypothesis test
]

print("\nAdditional test cases:")
for delta, r in test_cases:
    result = sp.pseudo_huber(delta, r)
    print(f"sp.pseudo_huber({delta:.2e}, {r}) = {result}")

# Show expected value calculation
print("\n" + "=" * 60)
print("Expected behavior:")
print("For small delta and r=1.0, pseudo_huber should return approximately |r| - delta ≈ 1.0")
print("The mathematical formula δ²(√(1 + (r/δ)²) - 1) is well-defined for all positive delta.")
```

<details>

<summary>
Output shows NaN for delta < 1e-190
</summary>
```
Testing scipy.special.pseudo_huber with small delta values:
============================================================
sp.pseudo_huber(1e-200, 1.0) = nan

Additional test cases:
sp.pseudo_huber(1.00e-100, 1.0) = 1.000000000000011e-100
sp.pseudo_huber(1.00e-150, 1.0) = 9.999999999999882e-151
sp.pseudo_huber(1.00e-190, 1.0) = nan
sp.pseudo_huber(1.00e-200, 1.0) = nan
sp.pseudo_huber(1.00e-250, 1.0) = nan
sp.pseudo_huber(1.00e-300, 1.0) = nan
sp.pseudo_huber(2.36e-203, 1.0) = nan

============================================================
Expected behavior:
For small delta and r=1.0, pseudo_huber should return approximately |r| - delta ≈ 1.0
The mathematical formula δ²(√(1 + (r/δ)²) - 1) is well-defined for all positive delta.
```
</details>

## Why This Is A Bug

The pseudo-Huber loss function is mathematically defined as δ²(√(1 + (r/δ)²) - 1) and should return finite values for all positive delta and finite r values. The function's documentation explicitly states it should be:

1. **Smooth and continuously differentiable everywhere** - The NaN output violates this fundamental property
2. **Convex** - Convex functions must be finite throughout their domain
3. **A robust loss function for optimization** - Returning NaN breaks gradient-based optimization algorithms

The bug occurs due to numerical overflow when computing (r/delta)² for small delta values:
- When delta < ~1e-190: (r/delta)² overflows to infinity
- sqrt(infinity) = infinity
- infinity - 1 = infinity
- delta² * infinity = 0 * infinity = NaN

The documentation makes no mention of restrictions on delta values and all examples work with delta >= 1.0. Users would reasonably expect the function to handle the full range of positive floating-point values, especially in scipy.special which is designed for numerical accuracy in scientific computing.

## Relevant Context

- The pseudo-Huber function is used in robust statistics and machine learning as a smooth alternative to the Huber loss
- Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pseudo_huber.html
- The function was added in scipy 0.15.0 for gradient-based optimization where smoothness is critical
- For comparison, the non-smooth `huber` function doesn't suffer from this issue
- The bug manifests at delta values around 1e-190, well within the range of valid Python floats (minimum positive normal float64 is ~2.2e-308)

## Proposed Fix

The implementation should detect when |r/delta| would cause overflow and switch to an asymptotically equivalent formula. For large |r/delta|, the pseudo-Huber loss approaches |r| - delta:

```diff
def pseudo_huber(delta, r):
+   # Avoid overflow for small delta by checking if (r/delta)^2 would overflow
+   # For float64, overflow occurs when |r/delta| > ~1e154
+   if np.abs(r) > delta * 1e150:
+       # Use asymptotic approximation: pseudo_huber ≈ |r| - delta
+       return np.abs(r) - delta
+
    # Original formula - safe when no overflow risk
    return delta**2 * (np.sqrt(1 + (r/delta)**2) - 1)
```

Alternatively, refactor to avoid explicitly computing (r/delta)²:

```diff
def pseudo_huber(delta, r):
+   # Compute in a numerically stable way
+   abs_r = np.abs(r)
+   if abs_r <= delta:
+       # When |r| <= delta, use standard formula (no overflow risk)
+       return delta**2 * (np.sqrt(1 + (r/delta)**2) - 1)
+   else:
+       # When |r| > delta, rearrange to avoid division overflow
+       # δ²(√(1 + (r/δ)²) - 1) = δ²((|r|/δ)√(1 + (δ/r)²) - 1)
+       #                       = δ|r|√(1 + (δ/r)²) - δ²
+       #                       ≈ |r| - δ²/(2|r|) for large |r/δ|
+       ratio_inv_sq = (delta/abs_r)**2
+       return delta * abs_r * (np.sqrt(1 + ratio_inv_sq) - delta/abs_r)
```