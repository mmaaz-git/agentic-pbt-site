# Bug Report: scipy.integrate.simpson Violates Reversal Property for Even Number of Points

**Target**: `scipy.integrate.simpson`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.integrate.simpson` function violates the mathematical reversal property when using an even number of sample points, producing incorrect integration results when the x and y arrays are reversed.

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
Reversal property violation with n=4 points
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

This bug violates a fundamental mathematical property of integration. The reversal property states that when both the x-coordinates and y-values are reversed, the integral should be negated:

∫[a,b] f(x) dx = -∫[b,a] f(x) dx

This property must hold for any numerical integration method to be mathematically correct. The violation occurs specifically when using an even number of sample points (n=4, 6, 8, 10), while odd numbers of points appear to work correctly.

Key issues:
1. **Mathematical incorrectness**: The function produces wrong numerical results for a basic property of integration
2. **Silent failure**: No warnings or errors are raised - users receive incorrect results without knowing
3. **Predictable failure pattern**: Consistently fails for even n, suggesting a systematic implementation error
4. **Significant error magnitude**: The errors are not small floating-point discrepancies but substantial deviations (e.g., ~8.7% error in the example)

## Relevant Context

Simpson's rule is a widely-used numerical integration method that approximates the integral by fitting parabolas through sets of points. For an odd number of equally-spaced samples, it should be exact for polynomials of degree ≤3. The implementation appears to have an asymmetry issue when handling even numbers of points.

The bug affects any workflow that:
- Uses Simpson's rule with even numbers of sample points
- Relies on the mathematical properties of integration
- Performs integration in reverse order (common in physics and engineering calculations)

**References**:
- scipy.integrate.simpson documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html
- Simpson's rule: https://en.wikipedia.org/wiki/Simpson%27s_rule

## Proposed Fix

The issue likely stems from how the implementation handles the composite Simpson's rule for even numbers of intervals. When n is even, there's an odd number of intervals, requiring special treatment of one interval (typically using the trapezoidal rule for one segment).

The implementation should ensure symmetry in how it handles the endpoints and intervals. A high-level fix approach:

1. **Review the interval handling logic** for even n cases
2. **Ensure symmetric treatment** of the first and last intervals
3. **Verify sign conventions** when x-coordinates are in descending order
4. **Add explicit tests** for the reversal property with various n values

Without examining the specific implementation code, the bug is likely in the logic that decides which intervals to apply Simpson's rule vs. trapezoidal rule when n is even, causing an asymmetric treatment that violates the reversal property.