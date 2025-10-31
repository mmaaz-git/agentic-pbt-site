# Bug Report: numpy.polynomial Power Operator Inconsistency with Coefficient Trimming

**Target**: `numpy.polynomial.Polynomial` and `numpy.polynomial.Chebyshev`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `**` power operator produces different coefficient arrays than repeated `*` multiplication when polynomial coefficients underflow to zero, violating the mathematical identity that `p**n` should equal `p*p*...*p` (n times).

## Property-Based Test

```python
import numpy as np
import numpy.polynomial as poly
from hypothesis import given, settings, strategies as st


@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3), min_size=1, max_size=5),
    st.integers(min_value=2, max_value=3)
)
def test_power_by_repeated_multiplication(coeffs, n):
    p = poly.Polynomial(coeffs)

    result_power = p ** n
    result_mult = p
    for _ in range(n - 1):
        result_mult = result_mult * p

    np.testing.assert_array_equal(result_power.coef, result_mult.coef)


# Run the test
if __name__ == "__main__":
    # First try with the specific failing input
    print("Testing specific failing input:")
    print("coeffs=[0.0, 1.1125369292536007e-308], n=2")

    # Manually test the failing case
    p = poly.Polynomial([0.0, 1.1125369292536007e-308])
    result_power = p ** 2
    result_mult = p * p
    try:
        np.testing.assert_array_equal(result_power.coef, result_mult.coef)
        print("Test passed")
    except AssertionError as e:
        print(f"Test failed with AssertionError: {e}")

    print("\nRunning property-based test with Hypothesis...")
    # Run the full property test
    from hypothesis import reproduce_failure
    test_power_by_repeated_multiplication()
```

<details>

<summary>
**Failing input**: `coeffs=[0.0, 1.1125369292536007e-308], n=2`
</summary>
```
Testing specific failing input:
coeffs=[0.0, 1.1125369292536007e-308], n=2
Test failed with AssertionError:
Arrays are not equal

(shapes (3,), (1,) mismatch)
 ACTUAL: array([0., 0., 0.])
 DESIRED: array([0.])

Running property-based test with Hypothesis...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 41, in <module>
    test_power_by_repeated_multiplication()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 7, in test_power_by_repeated_multiplication
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 19, in test_power_by_repeated_multiplication
    np.testing.assert_array_equal(result_power.coef, result_mult.coef)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header='Arrays are not equal',
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 803, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Arrays are not equal

(shapes (3,), (1,) mismatch)
 ACTUAL: array([0., 0., 0.])
 DESIRED: array([0.])
Falsifying example: test_power_by_repeated_multiplication(
    coeffs=[0.0, 1.7811007566526405e-172],
    n=2,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:793
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.polynomial as poly

# Test with minimal failing input from the bug report
p = poly.Polynomial([0.0, 1.1125369292536007e-308])

# Compare power operator vs repeated multiplication
result_power = p ** 2
result_mult = p * p

print("Testing with coefficients [0.0, 1.1125369292536007e-308]:")
print(f"p**2 coefficients: {result_power.coef}")
print(f"p*p coefficients: {result_mult.coef}")
print(f"Arrays equal: {np.array_equal(result_power.coef, result_mult.coef)}")
print()

# Also test with Chebyshev polynomials
c = poly.Chebyshev([0.0, 1.1125369292536007e-308])
result_power_c = c ** 2
result_mult_c = c * c

print("Testing Chebyshev with same coefficients:")
print(f"c**2 coefficients: {result_power_c.coef}")
print(f"c*c coefficients: {result_mult_c.coef}")
print(f"Arrays equal: {np.array_equal(result_power_c.coef, result_mult_c.coef)}")
```

<details>

<summary>
Output shows coefficient array mismatch
</summary>
```
Testing with coefficients [0.0, 1.1125369292536007e-308]:
p**2 coefficients: [0. 0. 0.]
p*p coefficients: [0.]
Arrays equal: False

Testing Chebyshev with same coefficients:
c**2 coefficients: [0. 0. 0.]
c*c coefficients: [0.]
Arrays equal: False
```
</details>

## Why This Is A Bug

This violates the fundamental algebraic property that `p^n = p × p × ... × p` (n times). When squaring the tiny coefficient `1.1125369292536007e-308`, it underflows to `0.0`, resulting in a polynomial with all zero coefficients. However, the two mathematically equivalent operations produce different representations:

1. **Multiplication operator (`*`)**: Correctly trims trailing zeros via `pu.trimseq()` in `polymul()`, producing the canonical form `[0.]`
2. **Power operator (`**`)**: Does not trim zeros after computation, preserving the full polynomial structure `[0., 0., 0.]`

This inconsistency breaks reasonable expectations:
- The mathematical identity `p**n == p*p*...*p` should hold at the representation level
- numpy's own testing utility `numpy.testing.assert_array_equal` fails on mathematically equivalent results
- Code comparing polynomial objects or relying on consistent array shapes will fail unexpectedly

While both polynomials evaluate to the same values (zero everywhere), the coefficient array inconsistency is a genuine bug that violates the principle of least surprise and internal consistency within the library.

## Relevant Context

The root cause is in the implementation differences between operators:

- `polymul()` in `/numpy/polynomial/polynomial.py:366` calls `pu.trimseq(ret)` to trim zeros
- `polypow()` in `/numpy/polynomial/polynomial.py:463` uses `pu._pow(np.convolve, c, pow, maxpower)` which doesn't trim
- The `_pow` helper in `/numpy/polynomial/polyutils.py` repeatedly calls the multiplication function but doesn't apply trimming to the final result

Interestingly, other polynomial classes (`Legendre`, `Hermite`, `HermiteE`, `Laguerre`) handle this correctly and produce consistent `[0.]` arrays for both operations. This shows the expected behavior is achievable and already implemented elsewhere in the library.

NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.html

Tested with NumPy version: 2.3.0

## Proposed Fix

```diff
--- a/numpy/polynomial/polyutils.py
+++ b/numpy/polynomial/polyutils.py
@@ -654,7 +654,7 @@ def _pow(mul_f, c, pow, maxpower):
         prd = c
         for i in range(2, power + 1):
             prd = mul_f(prd, c)
-        return prd
+        return trimseq(prd)
```

Alternatively, fix at the polynomial.py level:

```diff
--- a/numpy/polynomial/polynomial.py
+++ b/numpy/polynomial/polynomial.py
@@ -460,7 +460,8 @@ def polypow(c, pow, maxpower=None):
     """
     # note: this is more efficient than `pu._pow(polymul, c1, c2)`, as it
     # avoids calling `as_series` repeatedly
-    return pu._pow(np.convolve, c, pow, maxpower)
+    result = pu._pow(np.convolve, c, pow, maxpower)
+    return pu.trimseq(result)
```