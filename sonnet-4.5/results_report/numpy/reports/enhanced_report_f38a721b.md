# Bug Report: numpy.polynomial.Polynomial.__pow__ Inconsistent Coefficient Trimming

**Target**: `numpy.polynomial.Polynomial.__pow__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `__pow__` operator for Polynomial objects does not trim trailing zero coefficients, causing polynomials created via exponentiation to have different coefficient arrays than mathematically equivalent polynomials created via repeated multiplication, leading to unexpected equality check failures.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from numpy.polynomial import Polynomial


@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=6),
    st.integers(min_value=2, max_value=5)
)
def test_power_equals_repeated_multiplication(coeffs, n):
    p = Polynomial(coeffs)

    power_result = p ** n

    mult_result = Polynomial([1])
    for _ in range(n):
        mult_result = mult_result * p

    # Check if coefficient arrays have the same shape first
    if power_result.coef.shape != mult_result.coef.shape:
        raise AssertionError(
            f"Coefficient shapes differ: p**{n} has shape {power_result.coef.shape}, "
            f"p*...*p has shape {mult_result.coef.shape}. "
            f"Coeffs: {coeffs}, n: {n}"
        )

    assert np.allclose(power_result.coef, mult_result.coef, atol=1e-8), \
        f"p**{n} != p*...*p (repeated {n} times)"

# Run the test
test_power_equals_repeated_multiplication()
```

<details>

<summary>
**Failing input**: `coeffs=[0.0, 2.2250738585072014e-308], n=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 32, in <module>
    test_power_equals_repeated_multiplication()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 7, in test_power_equals_repeated_multiplication
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 22, in test_power_equals_repeated_multiplication
    raise AssertionError(
    ...<3 lines>...
    )
AssertionError: Coefficient shapes differ: p**2 has shape (3,), p*...*p has shape (1,). Coeffs: [0.0, 2.2250738585072014e-308], n: 2
Falsifying example: test_power_equals_repeated_multiplication(
    coeffs=[0.0, 2.2250738585072014e-308],
    n=2,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/63/hypo.py:22
```
</details>

## Reproducing the Bug

```python
from numpy.polynomial import Polynomial
import numpy as np

# Use the failing case found by Hypothesis
coeffs = [0.0, 2.2250738585072014e-308]
n = 2

print(f"Testing polynomial with coefficients: {coeffs}")
print(f"Raising to power: {n}")
print()

# Create the polynomial
p = Polynomial(coeffs)

# Compute p**2 using power operator
power_result = p ** n

# Compute p*p using multiplication
mult_result = Polynomial([1])
for _ in range(n):
    mult_result = mult_result * p

print("Original polynomial p:")
print(f"  Coefficients: {p.coef}")
print(f"  Shape: {p.coef.shape}")
print()

print(f"Result of p**{n}:")
print(f"  Coefficients: {power_result.coef}")
print(f"  Shape: {power_result.coef.shape}")
print()

print(f"Result of p*p (repeated multiplication):")
print(f"  Coefficients: {mult_result.coef}")
print(f"  Shape: {mult_result.coef.shape}")
print()

print("Comparison:")
print(f"  Shapes match? {power_result.coef.shape == mult_result.coef.shape}")
print(f"  p**{n} == p*p? {power_result == mult_result}")

# Show that the polynomials evaluate to the same values even though they're not "equal"
test_values = [0, 1, -1, 2, -2]
print("\nEvaluation at test points:")
for x in test_values:
    print(f"  At x={x:3}: p**{n}(x) = {power_result(x):.2e}, p*p(x) = {mult_result(x):.2e}")
```

<details>

<summary>
Output showing shape mismatch but identical evaluations
</summary>
```
Testing polynomial with coefficients: [0.0, 2.2250738585072014e-308]
Raising to power: 2

Original polynomial p:
  Coefficients: [0.00000000e+000 2.22507386e-308]
  Shape: (2,)

Result of p**2:
  Coefficients: [0. 0. 0.]
  Shape: (3,)

Result of p*p (repeated multiplication):
  Coefficients: [0.]
  Shape: (1,)

Comparison:
  Shapes match? False
  p**2 == p*p? False

Evaluation at test points:
  At x=  0: p**2(x) = 0.00e+00, p*p(x) = 0.00e+00
  At x=  1: p**2(x) = 0.00e+00, p*p(x) = 0.00e+00
  At x= -1: p**2(x) = 0.00e+00, p*p(x) = 0.00e+00
  At x=  2: p**2(x) = 0.00e+00, p*p(x) = 0.00e+00
  At x= -2: p**2(x) = 0.00e+00, p*p(x) = 0.00e+00
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical principle that `p^n = p × p × ... × p` (n times). The bug occurs because:

1. **Inconsistent trimming behavior**: The `polymul` function (used by `__mul__`) calls `pu.trimseq()` to remove trailing zeros after convolution, but `polypow` (used by `__pow__`) does not trim its result.

2. **Equality check failure**: The `__eq__` method in `ABCPolyBase` checks `self.coef.shape == other.coef.shape` (line 647 of `_polybase.py`), so even mathematically equivalent polynomials are considered unequal if their coefficient arrays have different shapes.

3. **Unexpected behavior**: Users expect `p**2` to equal `p*p`, but this fails for any polynomial with trailing zeros or near-zero coefficients that get rounded to zero during computation.

4. **Documentation contradiction**: The numpy polynomial documentation states that operations should maintain mathematical consistency, but this behavior violates that principle.

## Relevant Context

The root cause is in the implementation details:

- **`polymul` function** (`polynomial.py:363-366`): Uses `np.convolve` then calls `pu.trimseq(ret)` to remove trailing zeros
- **`polypow` function** (`polynomial.py:463`): Calls `pu._pow(np.convolve, c, pow, maxpower)` without trimming
- **`_pow` helper** (`polyutils.py:699-701`): Repeatedly multiplies using the provided multiplication function but doesn't trim the final result
- **`__pow__` method** (`_polybase.py:589-592`): Creates a new Polynomial with untrimmed coefficients

This affects all polynomial types that inherit from `ABCPolyBase` and use similar power/multiplication patterns.

## Proposed Fix

```diff
--- a/numpy/polynomial/_polybase.py
+++ b/numpy/polynomial/_polybase.py
@@ -589,7 +589,7 @@ class ABCPolyBase(abc.ABC):
     def __pow__(self, other):
         coef = self._pow(self.coef, other, maxpower=self.maxpower)
         res = self.__class__(coef, self.domain, self.window, self.symbol)
-        return res
+        return res.trim()

     def __radd__(self, other):
         try:
```

This ensures the power operator returns a trimmed polynomial, consistent with multiplication and other operations, maintaining mathematical consistency across the API.