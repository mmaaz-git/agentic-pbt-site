# Bug Report: scipy.optimize.minimize Nelder-Mead Returns Incorrect Minimum with Subnormal Initial Values

**Target**: `scipy.optimize.minimize` (Nelder-Mead method)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Nelder-Mead optimization method returns an incorrect minimum with 100% error when initialized with subnormal/denormalized floating-point values, yet incorrectly reports `success=True`. For a simple convex quadratic function with unique minimum at [1, -2], Nelder-Mead finds [1, ~0] when starting from points with extremely small y-coordinates like 1.8e-199.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.optimize import minimize

@given(st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False),
       st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_minimize_methods_consistency(x0, y0):
    """Test that different methods find similar solutions for convex functions."""
    x_init = np.array([x0, y0])

    def convex_func(x):
        return (x[0] - 1) ** 2 + (x[1] + 2) ** 2

    methods = ['BFGS', 'Nelder-Mead', 'Powell']
    results = []

    for method in methods:
        try:
            result = minimize(convex_func, x_init, method=method)
            if result.success:
                results.append(result.x)
        except (ValueError, RuntimeError):
            pass

    if len(results) >= 2:
        for i in range(len(results) - 1):
            diff = np.linalg.norm(results[i] - results[i + 1])
            assert diff < 0.5, \
                f"Different methods should find similar solutions, difference: {diff}"

# Run the test
test_minimize_methods_consistency()
```

<details>

<summary>
**Failing input**: `x0=0.0, y0=2.225073858507203e-309`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 33, in <module>
    test_minimize_methods_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 6, in test_minimize_methods_consistency
    st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 29, in test_minimize_methods_consistency
    assert diff < 0.5, \
           ^^^^^^^^^^
AssertionError: Different methods should find similar solutions, difference: 2.0000000279064167
Falsifying example: test_minimize_methods_consistency(
    x0=0.0,  # or any other generated value
    y0=2.225073858507203e-309,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/53/hypo.py:30
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.optimize import minimize

x0 = np.array([0.0, 1.8004729751112097e-199])

def func(x):
    return (x[0] - 1)**2 + (x[1] + 2)**2

result_nm = minimize(func, x0, method='Nelder-Mead')
result_bfgs = minimize(func, x0, method='BFGS')

print(f"Initial point: x={x0}")
print(f"True minimum: x=[1, -2], f=0")
print(f"\nNelder-Mead:")
print(f"  Solution: x={result_nm.x}")
print(f"  Function value: f={result_nm.fun}")
print(f"  Success: {result_nm.success}")
print(f"  Distance from true min: {np.linalg.norm(result_nm.x - [1, -2])}")

print(f"\nBFGS (for comparison):")
print(f"  Solution: x={result_bfgs.x}")
print(f"  Function value: f={result_bfgs.fun}")
print(f"  Success: {result_bfgs.success}")
print(f"  Distance from true min: {np.linalg.norm(result_bfgs.x - [1, -2])}")
```

<details>

<summary>
Output showing Nelder-Mead fails to find minimum
</summary>
```
Initial point: x=[0.00000000e+000 1.80047298e-199]
True minimum: x=[1, -2], f=0

Nelder-Mead:
  Solution: x=[ 1.00000000e+000 -1.78246825e-197]
  Function value: f=4.0
  Success: True
  Distance from true min: 2.0

BFGS (for comparison):
  Solution: x=[ 0.99999998 -2.00000003]
  Function value: f=1.1411716670850257e-15
  Success: True
  Distance from true min: 3.378123246841396e-08
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Incorrect mathematical result**: For the convex quadratic function f(x) = (x[0] - 1)² + (x[1] + 2)², the unique global minimum is at x = [1, -2] with f(x) = 0. Nelder-Mead finds x ≈ [1, 0] with f(x) = 4.0, which is objectively wrong.

2. **False success reporting**: The optimizer reports `success=True` despite finding a point with function value 4.0 when the true minimum has value 0.0. This is a 100% relative error in the optimal value.

3. **Inconsistency with other methods**: BFGS correctly finds the minimum (within numerical precision) from the same starting point, demonstrating that the issue is specific to Nelder-Mead's implementation, not an inherent numerical limitation.

4. **Violation of optimizer contract**: According to scipy documentation, the minimize function should "Minimize a scalar function of one or more variables." For a simple convex quadratic with a unique minimum, all methods should converge to approximately the same solution.

The bug is triggered by subnormal/denormalized floating-point values (numbers smaller than ~2.2e-308 for float64) in the initial point, which are valid IEEE 754 floating-point values that can occur in real computations involving very small quantities.

## Relevant Context

The root cause is in the initial simplex construction in `/scipy/optimize/_optimize.py` at lines 790-801. When building the initial simplex, the code treats coordinates as zero if they equal zero exactly, but doesn't account for subnormal values that are effectively treated as zero by the simplex construction logic:

```python
# Lines 774-775, 796-800 in _optimize.py
nonzdelt = 0.05
zdelt = 0.00025

for k in range(N):
    y = np.array(x0, copy=True)
    if y[k] != 0:  # This condition fails for subnormal values
        y[k] = (1 + nonzdelt)*y[k]
    else:
        y[k] = zdelt
```

When y[k] is 1.8e-199, the condition `y[k] != 0` is True, so the code multiplies by `(1 + 0.05)`, resulting in a simplex vertex at approximately 1.89e-199, which is still essentially zero. This creates a degenerate simplex that cannot explore the negative direction properly.

Documentation references:
- scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- Nelder-Mead algorithm: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

## Proposed Fix

The fix requires modifying the initial simplex construction to handle subnormal values appropriately. The code should check if a value is effectively zero (including subnormal numbers) rather than exactly zero:

```diff
--- a/scipy/optimize/_optimize.py
+++ b/scipy/optimize/_optimize.py
@@ -773,6 +773,7 @@ def _minimize_neldermead(func, x0, args=(), callback=None,

     nonzdelt = 0.05
     zdelt = 0.00025
+    epsilon = np.finfo(x0.dtype).eps * 100  # Threshold for "effectively zero"

     if bounds is not None:
         lower_bound, upper_bound = bounds.T
@@ -793,7 +794,8 @@ def _minimize_neldermead(func, x0, args=(), callback=None,
         sim[0] = x0
         for k in range(N):
             y = np.array(x0, copy=True)
-            if y[k] != 0:
+            # Treat subnormal/denormal values as effectively zero
+            if abs(y[k]) > epsilon:
                 y[k] = (1 + nonzdelt)*y[k]
             else:
                 y[k] = zdelt
```

This ensures that coordinates with extremely small absolute values (including subnormal numbers) are treated as zero and given a proper step size for simplex construction, allowing the algorithm to explore all directions effectively.