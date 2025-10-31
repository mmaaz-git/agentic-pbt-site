# Bug Report: scipy.optimize.ridder Incorrect Convergence Detection with Custom Tolerances

**Target**: `scipy.optimize.ridder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ridder` root-finding method incorrectly reports convergence failure when using custom tolerance parameters (xtol=1e-3, rtol=1e-3) despite finding roots to machine precision, exhibiting asymmetric behavior between positive and negative roots.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
from scipy.optimize import ridder

@given(
    st.floats(min_value=-10, max_value=-0.1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_ridder_converges_with_custom_tolerance(a, b):
    assume(abs(a - b) > 1e-6)

    def f(x):
        return x * x - 2.0

    fa, fb = f(a), f(b)
    assume(fa * fb < 0)

    result = ridder(f, a, b, xtol=1e-3, rtol=1e-3, full_output=True, disp=False)
    root, info = result

    assert info.converged, f"ridder failed to converge for interval [{a}, {b}]"
    assert abs(f(root)) < 1e-6, f"f(root) = {f(root)}, expected ~0"
```

<details>

<summary>
**Failing input**: `a=-2.0, b=1.0`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 25, in <module>
  |     test_ridder_converges_with_custom_tolerance()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 5, in test_ridder_converges_with_custom_tolerance
  |     st.floats(min_value=-10, max_value=-0.1, allow_nan=False, allow_infinity=False),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 21, in test_ridder_converges_with_custom_tolerance
    |     assert info.converged, f"ridder failed to converge for interval [{a}, {b}]"
    |            ^^^^^^^^^^^^^^
    | AssertionError: ridder failed to converge for interval [-2.0, 1.0]
    | Falsifying example: test_ridder_converges_with_custom_tolerance(
    |     a=-2.0,  # or any other generated value
    |     b=1.0,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 22, in test_ridder_converges_with_custom_tolerance
    |     assert abs(f(root)) < 1e-6, f"f(root) = {f(root)}, expected ~0"
    |            ^^^^^^^^^^^^^^^^^^^
    | AssertionError: f(root) = -0.00334556221064064, expected ~0
    | Falsifying example: test_ridder_converges_with_custom_tolerance(
    |     a=-1.0,
    |     b=2.0,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from scipy.optimize import ridder

def f(x):
    return x * x - 2.0

print("Test case: Finding root of f(x) = x² - 2 in [-2, 1]")
print("Expected root: -√2 ≈ -1.41421356...")

result = ridder(f, -2.0, 1.0, xtol=1e-3, rtol=1e-3, full_output=True, disp=False)
root, info = result

print(f"Converged: {info.converged}")
print(f"Iterations: {info.iterations}")
print(f"Root: {root:.15f}")
print(f"f(root): {f(root):.2e}")

print("\nComparison: Same function, interval [0, 2] (positive root)")
result2 = ridder(f, 0.0, 2.0, xtol=1e-3, rtol=1e-3, full_output=True)
root2, info2 = result2
print(f"Converged: {info2.converged}")
print(f"Iterations: {info2.iterations}")
print(f"Root: {root2:.15f}")
print(f"f(root): {f(root2):.2e}")
```

<details>

<summary>
ridder reports convergence failure despite finding root to machine precision
</summary>
```
Test case: Finding root of f(x) = x² - 2 in [-2, 1]
Expected root: -√2 ≈ -1.41421356...
Converged: False
Iterations: 100
Root: -1.414213562373095
f(root): -4.44e-16

Comparison: Same function, interval [0, 2] (positive root)
Converged: True
Iterations: 3
Root: 1.413193860008073
f(root): -2.88e-03
```
</details>

## Why This Is A Bug

The ridder method violates its documented convergence criterion when custom tolerances are specified. The function finds the negative root of x²-2 to machine precision (residual: -4.44e-16, essentially zero) but incorrectly reports `converged: False` after exhausting 100 iterations. In contrast, when finding the positive root of the same function with identical tolerance settings, it reports successful convergence after just 3 iterations despite achieving much worse precision (residual: -2.88e-03).

The documentation states that the convergence criterion should be satisfied when `np.isclose(x, x0, atol=xtol, rtol=rtol)` where x is the exact root and x0 is the computed root. For the negative root case, the computed root -1.414213562373095 differs from the exact root -√2 by only 2.22e-16, which is well within the specified tolerances (xtol=1e-3, rtol=1e-3). The criterion `abs(x - x0) <= xtol + rtol * abs(x0)` evaluates to `2.22e-16 <= 1e-3 + 1e-3 * 1.414 ≈ 2.41e-03`, which is clearly satisfied.

This asymmetric behavior between positive and negative roots indicates a bug in the convergence checking logic, likely in the underlying C implementation. The algorithm successfully finds the root but fails to recognize its achievement, causing unnecessary failures for users who require custom tolerances for their specific applications.

## Relevant Context

The bug manifests specifically when:
- Using custom tolerance parameters (particularly rtol=1e-3)
- Finding negative roots (positive roots work correctly)
- The function has roots on both sides of zero

The ridder implementation is in scipy/optimize/_zeros_py.py which calls into a C implementation. The Python wrapper is at: `/home/npc/.local/lib/python3.13/site-packages/scipy/optimize/_zeros_py.py`

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.ridder.html

The issue affects scipy version 1.16.2 and likely earlier versions. Users encountering this issue can work around it by:
1. Using default tolerances (omitting xtol and rtol parameters)
2. Using alternative root-finding methods like brentq or brenth
3. Only specifying xtol without rtol

## Proposed Fix

The bug appears to be in the convergence check logic within the C implementation. A workaround at the Python level could be to verify convergence independently after the algorithm completes:

```diff
--- a/scipy/optimize/_zeros_py.py
+++ b/scipy/optimize/_zeros_py.py
@@ -XXX,Y +XXX,Z @@ def ridder(f, a, b, xtol=2e-12, rtol=8.881784197001252e-16,
     if full_output:
         x, funcalls, iterations, converged = _zeros._ridder(f, a, b, xtol, rtol,
                                                               maxiter, args, full_output, disp)
+        # Additional convergence check for cases where C implementation fails
+        if not converged and iterations == maxiter:
+            # Check if we actually found a root despite non-convergence report
+            fx = f(x, *args)
+            if abs(fx) < max(xtol, rtol * abs(x)):
+                converged = True
         results = RootResults(root=x, iterations=iterations,
                                function_calls=funcalls, converged=converged)
         return x, results
```

However, the proper fix should be in the C implementation (`scipy/optimize/Zeros/ridder.c`) to correctly handle the convergence criterion for negative roots with custom rtol values.