# Bug Report: numpy.lib.scimath.power Returns NaN Imaginary Part for Extremely Small Negative Base with Negative Even Exponent

**Target**: `numpy.lib.scimath.power`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.lib.scimath.power()` returns a complex number with NaN imaginary part when given an extremely small negative base and a negative even integer exponent that causes overflow, instead of returning `(inf+0j)`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings, example
import numpy as np
import numpy.lib.scimath as scimath

@given(
    st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
@example(x=-2.0758172915594093e-87, p=-4.0)  # Known failing case
@settings(max_examples=500)
def test_scimath_power_general(x, p):
    assume(abs(x) > 1e-100)
    assume(abs(p) > 1e-10 and abs(p) < 100)

    result = scimath.power(x, p)

    # Check for NaN in the result
    if np.iscomplex(result):
        assert not np.isnan(result.real) and not np.isnan(result.imag), \
            f"power({x}, {p}) = {result} contains NaN"
    else:
        assert not np.isnan(result), \
            f"power({x}, {p}) = {result} is NaN"

if __name__ == "__main__":
    test_scimath_power_general()
```

<details>

<summary>
**Failing input**: `x=-2.0758172915594093e-87, p=-4.0`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py:491: RuntimeWarning: divide by zero encountered in power
  return nx.power(x, p)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py:491: RuntimeWarning: invalid value encountered in power
  return nx.power(x, p)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 26, in <module>
    test_scimath_power_general()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 6, in test_scimath_power_general
    st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 19, in test_scimath_power_general
    assert not np.isnan(result.real) and not np.isnan(result.imag), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: power(-2.0758172915594093e-87, -4.0) = (inf+nanj) contains NaN
Falsifying explicit example: test_scimath_power_general(
    x=-2.0758172915594093e-87,
    p=-4.0,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.lib.scimath as scimath

x = -2.0758172915594093e-87
p = -4.0

result = scimath.power(x, p)
print(f"scimath.power({x}, {p}) = {result}")
print(f"Result: {result}")
print(f"Imaginary part: {result.imag}")
print(f"Imaginary part is NaN: {np.isnan(result.imag)}")

print("\nComparison with other negative bases:")
for test_x in [-1.0, -1e-10, -1e-50]:
    test_result = scimath.power(test_x, -4.0)
    print(f"scimath.power({test_x}, -4.0) = {test_result}")

print("\nComparison with numpy.power (doesn't handle complex):")
try:
    np_result = np.power(x, p)
    print(f"numpy.power({x}, {p}) = {np_result}")
except:
    print(f"numpy.power({x}, {p}) raises an error for negative base")
```

<details>

<summary>
Output shows NaN in imaginary part for overflow case
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py:491: RuntimeWarning: divide by zero encountered in power
  return nx.power(x, p)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py:491: RuntimeWarning: invalid value encountered in power
  return nx.power(x, p)
/home/npc/pbt/agentic-pbt/worker_/34/repo.py:20: RuntimeWarning: overflow encountered in power
  np_result = np.power(x, p)
scimath.power(-2.0758172915594093e-87, -4.0) = (inf+nanj)
Result: (inf+nanj)
Imaginary part: nan
Imaginary part is NaN: True

Comparison with other negative bases:
scimath.power(-1.0, -4.0) = (1+0j)
scimath.power(-1e-10, -4.0) = (9.999999999999999e+39+0j)
scimath.power(-1e-50, -4.0) = (1e+200+0j)

Comparison with numpy.power (doesn't handle complex):
numpy.power(-2.0758172915594093e-87, -4.0) = inf
```
</details>

## Why This Is A Bug

This violates the expected behavior and documented contract of `numpy.lib.scimath.power` in several ways:

1. **Mathematical Incorrectness**: For any negative real number `x` and even integer power `p`, the result `x^p` is a positive real number. When `p` is negative and even, `x^p = 1/(x^|p|) = 1/(positive number) = positive real`. The result should be `(inf+0j)` when overflow occurs, not `(inf+nanj)`.

2. **Inconsistent Behavior**: The function correctly returns complex numbers with `0j` imaginary parts for all other negative bases with negative even exponents, including cases that result in finite values and other infinity cases (e.g., `-1e-50` raised to `-4.0` gives `(1e+200+0j)`).

3. **Contract Violation**: The docstring states: "If `x` contains negative values, the output is converted to the complex domain." This implies the output should be a valid complex number. A complex number with NaN imaginary part is not a properly formed complex value in the mathematical sense.

4. **Comparison with numpy.power**: The underlying `numpy.power` function correctly returns `inf` (not NaN) for this case when working with real numbers. The issue arises during the conversion to complex domain.

## Relevant Context

The bug occurs in the interaction between:
1. The `_fix_real_lt_zero` function (line 119-122 in `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/lib/_scimath_impl.py`) which converts negative inputs to complex
2. The underlying `numpy.power` function which produces `inf+nanj` when computing powers of very small complex numbers with negative exponents

The issue appears when:
- The base is negative and extremely small (around 1e-87 or smaller)
- The exponent is a negative even number (-4.0, -6.0, -8.0, etc.)
- The result overflows to infinity

Related numpy documentation: https://numpy.org/doc/stable/reference/generated/numpy.lib.scimath.power.html

## Proposed Fix

The fix requires special handling in the `power` function to detect when the result overflows to infinity for negative bases with even integer exponents, ensuring the imaginary part remains 0:

```diff
--- a/numpy/lib/_scimath_impl.py
+++ b/numpy/lib/_scimath_impl.py
@@ -488,7 +488,17 @@ def power(x, p):
     """
     x = _fix_real_lt_zero(x)
     p = _fix_int_lt_zero(p)
-    return nx.power(x, p)
+    result = nx.power(x, p)
+
+    # Fix NaN imaginary part for overflow cases with negative base and even exponent
+    if nx.iscomplexobj(result):
+        # Check if we have inf+nanj and the exponent is an even integer
+        if nx.isinf(result.real) and nx.isnan(result.imag):
+            # For even integer exponents, the result should be real (0j imaginary)
+            p_scalar = p if nx.isscalar(p) else p.flat[0]
+            if p_scalar % 2 == 0 and p_scalar == int(p_scalar):
+                result = nx.where(nx.isnan(result.imag), result.real + 0j, result)
+    return result
```