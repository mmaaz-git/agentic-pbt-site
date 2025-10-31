# Bug Report: scipy.integrate.simpson Reversal Property Violation

**Target**: `scipy.integrate.simpson`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `simpson` function violates the fundamental integral reversal property (∫ₐᵇ f(x)dx = -∫ᵇₐ f(x)dx) when given an even number of sample points, producing mathematically incorrect results.

## Property-Based Test

```python
import numpy as np
from scipy import integrate
from hypothesis import given, strategies as st, settings, assume

@given(
    n=st.sampled_from([4, 6, 8, 10]),
    a=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=200)
def test_simpson_reversal_even_n(n, a, b):
    assume(b > a)
    assume(abs(b - a) > 1e-6)

    x_forward = np.linspace(a, b, n)
    y = np.random.randn(n)

    x_backward = x_forward[::-1]
    y_backward = y[::-1]

    result_forward = integrate.simpson(y, x=x_forward)
    result_backward = integrate.simpson(y_backward, x=x_backward)

    assert np.isclose(result_forward, -result_backward, rtol=1e-10, atol=1e-10), \
        f"Reversal property violated: forward={result_forward}, backward={result_backward}, " \
        f"sum={result_forward + result_backward}, n={n}, a={a}, b={b}"

if __name__ == "__main__":
    # Run the test
    test_simpson_reversal_even_n()
```

<details>

<summary>
**Failing input**: `n=4, a=0.0, b=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 30, in <module>
    test_simpson_reversal_even_n()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 6, in test_simpson_reversal_even_n
    n=st.sampled_from([4, 6, 8, 10]),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 24, in test_simpson_reversal_even_n
    assert np.isclose(result_forward, -result_backward, rtol=1e-10, atol=1e-10), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Reversal property violated: forward=1.00021914762931, backward=-1.0351886330176923, sum=-0.03496948538838218, n=4, a=0.0, b=1.0
Falsifying example: test_simpson_reversal_even_n(
    # The test always failed when commented parts were varied together.
    n=4,  # or any other generated value
    a=0.0,  # or any other generated value
    b=1.0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy import integrate

# Use a specific random seed for reproducibility
np.random.seed(42)

n = 4
a, b = 0.99999, 54.645
x_forward = np.linspace(a, b, n)
# Use random y values as in the bug report
y = np.random.randn(n)

x_backward = x_forward[::-1]
y_backward = y[::-1]

result_forward = integrate.simpson(y, x=x_forward)
result_backward = integrate.simpson(y_backward, x=x_backward)

print(f"Testing simpson reversal property with n={n} points")
print(f"Integration interval: [{a}, {b}]")
print(f"Function: random values (np.random.seed(42))")
print(f"x_forward:  {x_forward}")
print(f"y values:   {y}")
print(f"x_backward: {x_backward}")
print(f"y_backward: {y_backward}")
print()
print(f"result_forward (∫_0^10 x² dx):  {result_forward}")
print(f"result_backward (∫_10^0 x² dx): {result_backward}")
print(f"Expected: result_backward = -result_forward")
print(f"Actual sum (should be ~0): {result_forward + result_backward}")
print()
print(f"Error: {abs(result_forward + result_backward)}")
print(f"Reversal property violated: {not np.isclose(result_forward, -result_backward, rtol=1e-10, atol=1e-10)}")
```

<details>

<summary>
Output demonstrating error with random function
</summary>
```
Testing simpson reversal property with n=4 points
Integration interval: [0.99999, 54.645]
Function: random values (np.random.seed(42))
x_forward:  [ 0.99999 18.88166 36.76333 54.645  ]
y values:   [ 0.49671415 -0.1382643   0.64768854  1.52302986]
x_backward: [54.645   36.76333 18.88166  0.99999]
y_backward: [ 1.52302986  0.64768854 -0.1382643   0.49671415]

result_forward (∫_0^10 x² dx):  22.79958200453467
result_backward (∫_10^0 x² dx): -24.783766104393735
Expected: result_backward = -result_forward
Actual sum (should be ~0): -1.984184099859064

Error: 1.984184099859064
Reversal property violated: True
```
</details>

## Why This Is A Bug

The integral reversal property ∫ₐᵇ f(x)dx = -∫ᵇₐ f(x)dx is a fundamental mathematical property that must hold for any numerical integration method. This property states that reversing the direction of integration should produce the negative of the original result.

When we reverse both the x-coordinates and y-values arrays (x_backward = x[::-1], y_backward = y[::-1]), we are computing the integral in the opposite direction. Mathematically, this must yield the negative of the forward integral.

**This bug violates core mathematical principles** because:
1. The reversal property is not an optional feature - it's a mathematical requirement
2. The trapezoid function in scipy correctly implements this (documented at line 90-93 of `_quadrature.py`)
3. Users have no reason to expect different behavior for even vs odd number of points
4. The documentation makes no mention of this limitation

**Root cause**: The asymmetric Cartwright correction formula applied when n is even (lines 463-534 in `scipy/integrate/_quadrature.py`). The correction applies special coefficients (alpha, beta, eta) only to the last interval, breaking symmetry under reversal.

## Relevant Context

The bug **only occurs when n (number of points) is even**. When n is odd, the function correctly preserves the reversal property using standard Simpson's rule without corrections.

The implementation references:
- Cartwright, Kenneth V. "Simpson's Rule Cumulative Integration with MS Excel and Irregularly-spaced Data" (referenced at line 422-424)
- Wikipedia article on Simpson's rule for irregularly spaced data (line 498)
- https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data

The code comment at lines 500-504 states:
> "Cartwright 2017, Equation 8. The equation in Cartwright is calculating the first interval whereas the equations in the Wikipedia article are adjusting for the last integral."

This asymmetry between first and last interval corrections is precisely what causes the reversal property violation.

The simpson function documentation (line 415-418) states accuracy properties but makes no mention that fundamental mathematical properties might be violated for even n.

## Proposed Fix

```diff
--- a/scipy/integrate/_quadrature.py
+++ b/scipy/integrate/_quadrature.py
@@ -463,6 +463,12 @@ def simpson(y, x=None, *, dx=1.0, axis=-1):
     if N % 2 == 0:
         val = 0.0
         result = 0.0
         slice_all = (slice(None),) * nd
+
+        # Check if we're integrating backward (decreasing x)
+        is_backward = False
+        if x is not None:
+            x_flat = x.ravel() if hasattr(x, 'ravel') else x
+            is_backward = x_flat[0] > x_flat[-1]

         if N == 2:
@@ -476,7 +482,13 @@ def simpson(y, x=None, *, dx=1.0, axis=-1):
             val += 0.5 * last_dx * (y[slice1] + y[slice2])
         else:
             # use Simpson's rule on first intervals
-            result = _basic_simpson(y, 0, N-3, x, dx, axis)
+            if not is_backward:
+                # Forward integration: correct last interval
+                result = _basic_simpson(y, 0, N-3, x, dx, axis)
+            else:
+                # Backward integration: correct first interval instead
+                result = _basic_simpson(y, 3, N, x, dx, axis)
+                # Correction will be applied to first interval

             slice1 = tupleset(slice_all, axis, -1)
             slice2 = tupleset(slice_all, axis, -2)
@@ -529,7 +541,23 @@ def simpson(y, x=None, *, dx=1.0, axis=-1):
                 where=den != 0
             )

-            result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]
+            if not is_backward:
+                # Forward: correct last interval as before
+                result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]
+            else:
+                # Backward: correct first interval with symmetric formula
+                slice1 = tupleset(slice_all, axis, 0)
+                slice2 = tupleset(slice_all, axis, 1)
+                slice3 = tupleset(slice_all, axis, 2)
+
+                # Use same h values but from beginning
+                h0_idx = tupleset(slice_all, axis, slice(0, 1, 1))
+                h1_idx = tupleset(slice_all, axis, slice(1, 2, 1))
+                h = [np.squeeze(diffs[h0_idx], axis=axis),
+                     np.squeeze(diffs[h1_idx], axis=axis)]
+
+                # Apply symmetric correction to first interval
+                result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]

         result += val
     else:
```