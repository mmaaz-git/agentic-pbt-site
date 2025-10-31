# Bug Report: scipy.integrate.simpson Produces Mathematically Incorrect Results with Duplicate X Values

**Target**: `scipy.integrate.simpson`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.integrate.simpson` silently produces mathematically incorrect integration results when the input array `x` contains duplicate values, while the related function `cumulative_simpson` correctly validates and rejects such inputs with a ValueError.

## Property-Based Test

```python
import numpy as np
import scipy.integrate as integrate
from hypothesis import given, strategies as st, settings, assume
import pytest


@given(
    n_points=st.integers(min_value=4, max_value=10),
    dup_idx=st.integers(min_value=0, max_value=8)
)
@settings(max_examples=100)
def test_simpson_with_duplicate_x_values(n_points, dup_idx):
    """
    Property test: simpson produces incorrect results when x has duplicate values.

    This test creates an array where one x value is duplicated, creating a
    zero-width segment that should contribute 0 to the integral.
    """
    assume(dup_idx < n_points - 1)

    x = np.linspace(0, 1, n_points)
    x[dup_idx + 1] = x[dup_idx]
    y = x.copy()

    result = integrate.simpson(y, x=x)
    expected = 0.5

    if not np.isclose(result, expected, rtol=0.01):
        pytest.fail(f"simpson gives wrong result with duplicate x: {result} != {expected}")
```

<details>

<summary>
**Failing input**: `n_points=4, dup_idx=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 34, in <module>
    test_simpson_with_duplicate_x_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 8, in test_simpson_with_duplicate_x_values
    n_points=st.integers(min_value=4, max_value=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 29, in test_simpson_with_duplicate_x_values
    pytest.fail(f"simpson gives wrong result with duplicate x: {result} != {expected}")
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: simpson gives wrong result with duplicate x: 0.42592592592592593 != 0.5
Falsifying example: test_simpson_with_duplicate_x_values(
    n_points=4,
    dup_idx=0,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.integrate as integrate

# Test case from the bug report
x = np.array([0.0, 1.0, 1.0, 2.0])
y = np.array([0.0, 1.0, 1.0, 2.0])

result = integrate.simpson(y, x=x)

print(f"simpson result: {result}")
print(f"Expected: 2.0")
print()
print(f"Explanation: When integrating y=x from 0 to 2, the mathematical result")
print(f"should be ∫₀² x dx = x²/2 |₀² = 4/2 - 0/2 = 2.0")
print()
print(f"The duplicate x value at indices 1 and 2 (both equal to 1.0)")
print(f"represents a zero-width segment that should contribute 0 to the integral.")
print(f"However, simpson produces the result {result} instead of the correct value 2.0.")
```

<details>

<summary>
simpson returns incorrect mathematical result
</summary>
```
simpson result: 1.0
Expected: 2.0

Explanation: When integrating y=x from 0 to 2, the mathematical result
should be ∫₀² x dx = x²/2 |₀² = 4/2 - 0/2 = 2.0

The duplicate x value at indices 1 and 2 (both equal to 1.0)
represents a zero-width segment that should contribute 0 to the integral.
However, simpson produces the result 1.0 instead of the correct value 2.0.
```
</details>

## Why This Is A Bug

This violates expected mathematical behavior in multiple ways:

1. **Mathematical Incorrectness**: When integrating y=x from 0 to 2, the correct result is ∫₀² x dx = x²/2 |₀² = 2.0. The function returns 1.0, which is mathematically wrong by a factor of 2.

2. **Silent Failure**: The function accepts arrays with duplicate x values without any warning or error, then produces incorrect results. This is worse than raising an error because incorrect calculations can propagate unnoticed through scientific computations.

3. **Inconsistent API Behavior**: The closely related function `cumulative_simpson` in the same module correctly validates input and raises `ValueError: Input x must be strictly increasing.` for identical input. This inconsistency violates user expectations of uniform behavior across related integration functions.

4. **Violates Simpson's Rule Assumptions**: Simpson's rule requires proper intervals between points. When x[i] = x[i+1], the interval width is zero, which leads to division by zero or undefined behavior in the underlying mathematics. The implementation uses `np.true_divide` with `where` conditions to avoid runtime errors but produces meaningless results.

5. **Documentation Inconsistency**: While `cumulative_simpson` explicitly documents "x must also be strictly increasing along axis", `simpson` has no such documentation, yet both functions share the same mathematical requirements.

## Relevant Context

The bug occurs in the `_basic_simpson` helper function at scipy/integrate/_quadrature.py. When x values are provided, the function calculates spacing using `h = np.diff(x, axis=axis)` and then performs divisions like `h0/h1`. When duplicate x values exist, some h values become 0, leading to division operations that are handled with `np.true_divide(..., where=h1 != 0)` to avoid runtime errors, but this produces mathematically incorrect results.

In contrast, `cumulative_simpson` (same module) includes proper validation:
```python
dx = np.diff(x, axis=-1)
if np.any(dx <= 0):
    raise ValueError("Input x must be strictly increasing.")
```

Documentation references:
- scipy.integrate.simpson: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html
- scipy.integrate.cumulative_simpson: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_simpson.html

Source code location: scipy/integrate/_quadrature.py

## Proposed Fix

Add the same validation that `cumulative_simpson` uses to ensure x is strictly increasing:

```diff
--- a/scipy/integrate/_quadrature.py
+++ b/scipy/integrate/_quadrature.py
@@ -610,6 +610,11 @@ def simpson(y, x=None, *, dx=1.0, axis=-1):
         if x.shape[axis] != N:
             raise ValueError("If given, length of x along axis must be the "
                              "same as y.")
+
+        # Validate that x is strictly increasing (same as cumulative_simpson)
+        dx_check = np.diff(x, axis=axis)
+        if np.any(dx_check <= 0):
+            raise ValueError("Input x must be strictly increasing.")

     if N % 2 == 0:
         val = 0.0
```