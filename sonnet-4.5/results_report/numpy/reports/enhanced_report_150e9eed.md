# Bug Report: numpy.lib.scimath.power Returns NaN for Small Negative Numbers with Negative Even Powers

**Target**: `numpy.lib.scimath.power`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scimath.power` function returns a complex number with NaN in the imaginary part (`inf+nanj`) when raising small negative numbers (magnitude < ~1e-155) to negative even integer powers, instead of returning the mathematically correct result of positive infinity.

## Property-Based Test

```python
import numpy as np
import numpy.lib.scimath as scimath
from hypothesis import given, strategies as st, settings

@given(
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-5, max_value=5)
)
@settings(max_examples=1000, deadline=None)
def test_power_definition(x, n):
    """Test that scimath.power never returns NaN for valid inputs."""
    result = scimath.power(x, n)

    # Check if result contains NaN
    if np.isscalar(result):
        has_nan = np.isnan(result)
    else:
        has_nan = np.any(np.isnan(result))

    # The assertion that fails
    assert not has_nan, f"scimath.power({x}, {n}) returned {result} which contains NaN"

if __name__ == "__main__":
    # Run the test
    print("Running Hypothesis test for scimath.power...")
    print("Testing that scimath.power never returns NaN for valid (non-NaN, non-inf) inputs...")
    print()

    try:
        test_power_definition()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error:")
        print(f"  {e}")
        print()
        print("This confirms the bug: scimath.power returns NaN for certain valid inputs.")
        print()

        # Show the specific failing case
        x = -9.499558537778752e-188
        n = -2
        result = scimath.power(x, n)
        print(f"Minimal failing example:")
        print(f"  x = {x}")
        print(f"  n = {n}")
        print(f"  scimath.power(x, n) = {result}")
        print(f"  Contains NaN: {np.isnan(result)}")
        print()
        print("Expected: A valid complex or real number (likely inf or inf+0j)")
        print("Actual: inf+nanj (complex with NaN in imaginary part)")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `x = -7.94814457938965e-72, n = -5` (and others)
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py:491: RuntimeWarning: divide by zero encountered in power
  return nx.power(x, p)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py:491: RuntimeWarning: invalid value encountered in power
  return nx.power(x, p)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py:491: RuntimeWarning: overflow encountered in power
  return nx.power(x, p)
Running Hypothesis test for scimath.power...
Testing that scimath.power never returns NaN for valid (non-NaN, non-inf) inputs...

Test failed with assertion error:
  scimath.power(-7.94814457938965e-72, -5) returned (inf+nanj) which contains NaN

This confirms the bug: scimath.power returns NaN for certain valid inputs.

Minimal failing example:
  x = -9.499558537778752e-188
  n = -2
  scimath.power(x, n) = (inf+nanj)
  Contains NaN: True

Expected: A valid complex or real number (likely inf or inf+0j)
Actual: inf+nanj (complex with NaN in imaginary part)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.lib.scimath as scimath

# Test case from the bug report
x = -9.499558537778752e-188
n = -2
result = scimath.power(x, n)

print(f'scimath.power({x}, {n}) = {result}')
print(f'Has NaN in result: {np.isnan(result)}')
print(f'Result type: {type(result)}')
print(f'Result dtype: {result.dtype if hasattr(result, "dtype") else "N/A"}')

# Let's also test with simpler values
print("\nTesting with -1e-200:")
x2 = -1e-200
result2 = scimath.power(x2, -2)
print(f'scimath.power({x2}, {n}) = {result2}')
print(f'Has NaN: {np.isnan(result2)}')

# For comparison, let's see what regular numpy.power does
print("\nComparing with numpy.power:")
try:
    numpy_result = np.power(x2, -2)
    print(f'np.power({x2}, {n}) = {numpy_result}')
except Exception as e:
    print(f'np.power raised exception: {e}')

# Let's test the mathematical expectation
print("\nMathematical expectation:")
print(f"(-1e-200)^(-2) = 1/((-1e-200)^2) = 1/(1e-400) = 1e+400 = inf")
print(f"Since negative^even = positive, result should be positive infinity")

# Let's trace through what's happening step by step
print("\nStep-by-step trace:")
print(f"1. Input: x={x2}, p={n}")

# Convert to complex as scimath.power does
x_complex = np.asarray(x2, dtype=complex)
print(f"2. After _fix_real_lt_zero: x={x_complex}")

# Square it
squared = x_complex ** 2
print(f"3. x^2 = {squared}")

# Take reciprocal
if squared != 0:
    reciprocal = 1 / squared
    print(f"4. 1/(x^2) = {reciprocal}")
else:
    print(f"4. 1/(x^2) = division by {squared}")

# Check if the issue happens with other small negative values
print("\nTesting threshold for NaN occurrence:")
test_values = [-1e-150, -1e-155, -1e-160, -1e-170, -1e-180, -1e-190, -1e-200]
for val in test_values:
    res = scimath.power(val, -2)
    has_nan = np.isnan(res)
    print(f"scimath.power({val:e}, -2) = {res}, has NaN: {has_nan}")
```

<details>

<summary>
Output showing the bug and its threshold
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py:491: RuntimeWarning: divide by zero encountered in power
  return nx.power(x, p)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py:491: RuntimeWarning: invalid value encountered in power
  return nx.power(x, p)
/home/npc/pbt/agentic-pbt/worker_/62/repo.py:24: RuntimeWarning: overflow encountered in power
  numpy_result = np.power(x2, -2)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py:491: RuntimeWarning: overflow encountered in power
  return nx.power(x, p)
scimath.power(-9.499558537778752e-188, -2) = (inf+nanj)
Has NaN in result: True
Result type: <class 'numpy.complex128'>
Result dtype: complex128

Testing with -1e-200:
scimath.power(-1e-200, -2) = (inf+nanj)
Has NaN: True

Comparing with numpy.power:
np.power(-1e-200, -2) = inf

Mathematical expectation:
(-1e-200)^(-2) = 1/((-1e-200)^2) = 1/(1e-400) = 1e+400 = inf
Since negative^even = positive, result should be positive infinity

Step-by-step trace:
1. Input: x=-1e-200, p=-2
2. After _fix_real_lt_zero: x=(-1e-200+0j)
3. x^2 = -0j
4. 1/(x^2) = division by -0j

Testing threshold for NaN occurrence:
scimath.power(-1.000000e-150, -2) = (9.999999999999999e+299+0j), has NaN: False
scimath.power(-1.000000e-155, -2) = (inf+nanj), has NaN: True
scimath.power(-1.000000e-160, -2) = (inf+nanj), has NaN: True
scimath.power(-1.000000e-170, -2) = (inf+nanj), has NaN: True
scimath.power(-1.000000e-180, -2) = (inf+nanj), has NaN: True
scimath.power(-1.000000e-190, -2) = (inf+nanj), has NaN: True
scimath.power(-1.000000e-200, -2) = (inf+nanj), has NaN: True
```
</details>

## Why This Is A Bug

The `scimath.power` function is specifically designed to handle negative bases by converting to the complex domain, as stated in its documentation. However, it produces mathematically incorrect results for certain edge cases.

**Mathematical Correctness**: When raising a negative number to an even power, the result is always positive and real. For example:
- `(-1e-200)^(-2) = 1 / ((-1e-200)^2) = 1 / (1e-400) = 1e+400 = inf`
- The result should be positive infinity, not `inf+nanj`

**Documentation Contract Violation**: The function's own documentation demonstrates that even powers of negative numbers should have valid imaginary parts:
```python
>>> np.emath.power([-2, 4], 2)
array([ 4.-0.j, 16.+0.j])
```
The documentation shows `-0.j` or `+0.j` in the imaginary part, never NaN.

**Root Cause Analysis**: The bug occurs due to floating-point underflow:
1. `scimath.power` converts negative inputs to complex: `-1e-200` becomes `(-1e-200+0j)`
2. Squaring this underflows to complex negative zero: `(-1e-200+0j)^2 = -0j`
3. Taking the reciprocal `1/(-0j)` produces `inf+nanj` in numpy's complex arithmetic

**Impact**: NaN values propagate through calculations and can silently corrupt results in scientific computing applications. While the affected range (numbers with magnitude < ~1e-155) is extreme, the function should handle all valid inputs correctly.

## Relevant Context

- **Comparison with numpy.power**: Regular `numpy.power(-1e-200, -2)` correctly returns `inf` without any NaN
- **Threshold**: The bug occurs for negative numbers with magnitude smaller than approximately 1e-155
- **Function Purpose**: The `scimath` module exists to provide "mathematically valid answers in the complex plane" according to its module docstring
- **Related Functions**: Other scimath functions like `sqrt`, `log`, etc. correctly handle their edge cases
- **Documentation**: numpy.lib.scimath.power documentation at https://numpy.org/doc/stable/reference/generated/numpy.lib.scimath.power.html

## Proposed Fix

The issue can be fixed by detecting when the power is an even integer and avoiding the complex conversion for negative bases in those cases, since the result will always be real and positive:

```diff
--- a/numpy/lib/_scimath_impl.py
+++ b/numpy/lib/_scimath_impl.py
@@ -486,6 +486,15 @@ def power(x, p):
     array([ 4, 256])

     """
+    # For even integer powers, negative bases produce positive real results
+    # Avoid complex conversion to prevent inf+nanj in underflow cases
+    p_arr = asarray(p)
+    x_arr = asarray(x)
+    if (np.issubdtype(p_arr.dtype, np.integer) and  # integer power
+        np.all(p_arr % 2 == 0) and                   # even power
+        np.any(x_arr < 0)):                          # has negative values
+        # Use regular power which handles this correctly
+        return nx.power(x, p)
+
     x = _fix_real_lt_zero(x)
     p = _fix_int_lt_zero(p)
     return nx.power(x, p)
```