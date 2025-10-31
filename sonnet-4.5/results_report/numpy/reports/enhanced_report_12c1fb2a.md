# Bug Report: numpy.polynomial.Polynomial.roots() Returns Non-Root Values for Polynomials with Tiny Leading Coefficients

**Target**: `numpy.polynomial.Polynomial.roots()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The roots() method returns values that don't satisfy the fundamental mathematical property p(root) ≈ 0 when polynomials have tiny leading coefficients, instead returning numerically unstable values where p(root) can equal 1.0 or other non-zero values.

## Property-Based Test

```python
#!/usr/bin/env python3
import numpy as np
import numpy.polynomial as np_poly
from hypothesis import assume, given, settings, strategies as st


@settings(max_examples=1000)
@given(
    coef=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
        min_size=2,
        max_size=6
    )
)
def test_polynomial_roots_are_valid(coef):
    p = np_poly.Polynomial(coef)

    assume(p.degree() >= 1)

    roots = p.roots()

    assume(not np.any(np.isnan(roots)))
    assume(not np.any(np.isinf(roots)))

    for root in roots:
        value = abs(p(root))
        assert value < 1e-6, f'p({root}) = {p(root)}, expected ~0'


if __name__ == "__main__":
    # Run the test
    test_polynomial_roots_are_valid()
```

<details>

<summary>
**Failing input**: `coef=[1.0, 1.0, 5.176475981674391e-60]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 32, in <module>
    test_polynomial_roots_are_valid()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 8, in test_polynomial_roots_are_valid
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 27, in test_polynomial_roots_are_valid
    assert value < 1e-6, f'p({root}) = {p(root)}, expected ~0'
           ^^^^^^^^^^^^
AssertionError: p(-1.9318161690311532e+59) = 1.0, expected ~0
Falsifying example: test_polynomial_roots_are_valid(
    coef=[1.0, 1.0, 5.176475981674391e-60],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
import numpy as np
import numpy.polynomial as np_poly

# Test case from bug report with tiny leading coefficient
print("Test Case 1: Polynomial with tiny leading coefficient")
print("=" * 60)
coef = [1.0, 1.0, 3.9968426114653685e-66]
p = np_poly.Polynomial(coef)

print(f"Polynomial coefficients: {coef}")
print(f"Polynomial degree: {p.degree()}")
print()

# Compute roots
roots = p.roots()
print(f"Computed roots: {roots}")
print()

# Check if these are actually roots
print("Verification - evaluating p(root) for each root:")
for i, root in enumerate(roots):
    value = p(root)
    print(f"  p(roots[{i}]) = p({root:.6e}) = {value}")
print()

# Check the actual root (should be near -1 for this polynomial)
print("Checking actual root at x = -1.0:")
print(f"  p(-1.0) = {p(-1.0)}")
print()

# Second test case from bug report
print("Test Case 2: Another polynomial with tiny coefficient")
print("=" * 60)
coef2 = [0.0, 1.0, 3.254353641323301e-273]
p2 = np_poly.Polynomial(coef2)

print(f"Polynomial coefficients: {coef2}")
print(f"Polynomial degree: {p2.degree()}")
print()

# Compute roots
roots2 = p2.roots()
print(f"Computed roots: {roots2}")
print()

# Check if these are actually roots
print("Verification - evaluating p(root) for each root:")
for i, root in enumerate(roots2):
    value = p2(root)
    print(f"  p(roots[{i}]) = p({root:.6e}) = {value}")
```

<details>

<summary>
Output showing roots() returns values where p(root) ≠ 0
</summary>
```
Test Case 1: Polynomial with tiny leading coefficient
============================================================
Polynomial coefficients: [1.0, 1.0, 3.9968426114653685e-66]
Polynomial degree: 2

Computed roots: [-2.50197493e+65  0.00000000e+00]

Verification - evaluating p(root) for each root:
  p(roots[0]) = p(-2.501975e+65) = 1.0
  p(roots[1]) = p(0.000000e+00) = 1.0

Checking actual root at x = -1.0:
  p(-1.0) = 0.0

Test Case 2: Another polynomial with tiny coefficient
============================================================
Polynomial coefficients: [0.0, 1.0, 3.254353641323301e-273]
Polynomial degree: 2

Computed roots: [-3.0728068e+272  0.0000000e+000]

Verification - evaluating p(root) for each root:
  p(roots[0]) = p(-3.072807e+272) = -3.4115008600408656e+256
  p(roots[1]) = p(0.000000e+00) = 0.0
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical definition of polynomial roots. By definition, a root r of polynomial p must satisfy p(r) = 0. The numpy.polynomial.Polynomial.roots() documentation states it returns "the roots of the series polynomial", which implicitly promises values where the polynomial evaluates to zero.

In the test cases:
1. For polynomial `1.0 + 1.0*x + 3.9968426114653685e-66*x^2`, the actual root is at x = -1.0 (where p(-1) = 0), but roots() returns x = -2.50197493e+65 where p(x) = 1.0
2. The returned "roots" fail the basic verification test: evaluating the polynomial at these points yields non-zero values (1.0, -3.41e+256, etc.)
3. The issue stems from numerical instability when the companion matrix method encounters coefficients spanning extreme ranges (from 1.0 to 1e-66 or smaller)

While the documentation mentions accuracy degradation for roots outside the domain, it does not warn that the method can return completely invalid values that aren't roots at all. This silent failure is particularly problematic as users may not realize they're getting mathematically incorrect results.

## Relevant Context

The root cause is numerical instability in the companion matrix eigenvalue method used internally by numpy. When polynomial coefficients span many orders of magnitude (e.g., from 1.0 to 1e-66), the companion matrix construction leads to overflow/underflow issues. The eigenvalues computed from such matrices become meaningless.

This is a known challenge in numerical computing - Wilkinson's polynomial famously demonstrated extreme sensitivity to coefficient perturbations. Modern polynomial root-finding algorithms often include coefficient trimming or scaling to handle such cases.

Tiny coefficients can arise naturally from:
- Subtraction of nearly equal floating-point values
- Accumulation of round-off errors in iterative algorithms
- Physical simulations where different terms have vastly different scales

Documentation references:
- numpy.polynomial.Polynomial.roots(): https://numpy.org/doc/stable/reference/generated/numpy.polynomial.Polynomial.roots.html
- The method uses numpy.polynomial.polynomial.polyroots internally, which constructs a companion matrix

## Proposed Fix

The roots() method should trim negligible trailing coefficients before computing roots to avoid numerical instability. This is a standard technique in numerical polynomial algorithms.

```diff
--- a/numpy/polynomial/_polybase.py
+++ b/numpy/polynomial/_polybase.py
@@ -xxx,xxx +xxx,xxx @@ class ABCPolyBase:
     def roots(self):
         """Return the roots of the series polynomial."""
+        # Trim tiny trailing coefficients that cause numerical instability
+        # Use machine epsilon scaled by coefficient magnitude as tolerance
+        import numpy.polynomial.polyutils as pu
+        trimmed_coef = pu.trimcoef(self.coef, tol=np.finfo(float).eps * 100)
-        roots = self._roots(self.coef)
+        roots = self._roots(trimmed_coef)
         return pu.mapdomain(roots, self.window, self.domain)
```

Alternatively, the method could raise a warning when coefficient ranges exceed safe numerical limits, alerting users to potential accuracy issues.