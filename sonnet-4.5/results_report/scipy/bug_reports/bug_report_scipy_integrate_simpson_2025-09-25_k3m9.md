# Bug Report: scipy.integrate.simpson Produces Incorrect Results with Duplicate X Values

**Target**: `scipy.integrate.simpson`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.integrate.simpson` produces mathematically incorrect integration results when the input array `x` contains duplicate values (non-strictly-increasing sequences). The function silently accepts such inputs but computes wrong results, whereas the related function `cumulative_simpson` correctly validates and rejects non-strictly-increasing x values.

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

**Failing input**: `n_points=4, dup_idx=0`

## Reproducing the Bug

```python
import numpy as np
import scipy.integrate as integrate

x = np.array([0.0, 1.0, 1.0, 2.0])
y = np.array([0.0, 1.0, 1.0, 2.0])

result = integrate.simpson(y, x=x)

print(f"simpson result: {result}")
print(f"Expected: 2.0")
```

Output:
```
simpson result: 1.0
Expected: 2.0
```

## Why This Is A Bug

When integrating `y = x` from 0 to 2, the mathematical result should be `∫₀² x dx = x²/2 |₀² = 2.0`.

The duplicate x value at indices 1 and 2 (both equal to 1.0) represents a zero-width segment that should contribute 0 to the integral. However, `simpson` produces the result 1.0 instead of the correct value 2.0.

This is a **Logic bug** because:
1. The function produces mathematically incorrect results for valid numerical inputs
2. The behavior is inconsistent with the related function `cumulative_simpson`, which explicitly validates that x must be strictly increasing and raises `ValueError: Input x must be strictly increasing.` for the same input

This is also a **Contract bug** because:
1. The documentation for `simpson` does not specify any requirement that x must be strictly increasing
2. The documentation for `cumulative_simpson` explicitly states "x must also be strictly increasing along `axis`"
3. Users would reasonably expect either consistent behavior between these related functions OR clear documentation of the difference

## Fix

The bug can be fixed by adding validation to check that x is strictly increasing, similar to what `cumulative_simpson` does. Looking at the scipy source code, `cumulative_simpson` validates x values before processing.

Recommended fix:

```diff
diff --git a/scipy/integrate/_quadrature.py b/scipy/integrate/_quadrature.py
index 1234567..abcdefg 100644
--- a/scipy/integrate/_quadrature.py
+++ b/scipy/integrate/_quadrature.py
@@ -XXX,6 +XXX,11 @@ def simpson(y, x=None, *, dx=1.0, axis=-1):
         if x.shape[axis] != N:
             raise ValueError("If given, length of x along axis must be the "
                              "same as y.")
+
+        # Validate that x is strictly increasing
+        if x is not None:
+            if not np.all(np.diff(x, axis=axis) > 0):
+                raise ValueError("Input x must be strictly increasing.")

     if N % 2 == 0:
         val = 0.0
```

Alternatively, if the behavior with duplicate x values is intentional, the documentation should be updated to:
1. Explicitly state that x may contain duplicate values
2. Explain what the mathematical interpretation of the result is in such cases
3. Provide an example demonstrating this behavior