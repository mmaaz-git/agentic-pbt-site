# Bug Report: scipy.special.boxcox Numerical Instability and Silent Infinity Returns

**Target**: `scipy.special.boxcox` / `scipy.special.inv_boxcox`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Box-Cox inverse transformation `scipy.special.inv_boxcox` returns infinity or produces significant numerical errors when lambda is very negative (< -6) and x is moderately large (> 20), violating the mathematical property that the transformation should be invertible for all valid inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np
from scipy import special

@given(
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False)
)
def test_boxcox_inv_boxcox_roundtrip(lmbda, x):
    y = special.boxcox(x, lmbda)
    assume(not np.isinf(y) and not np.isnan(y))
    x_recovered = special.inv_boxcox(y, lmbda)
    assert np.isclose(x, x_recovered, rtol=1e-6, atol=1e-6), f"Failed for lmbda={lmbda}, x={x}: recovered {x_recovered}"

if __name__ == "__main__":
    test_boxcox_inv_boxcox_roundtrip()
```

<details>

<summary>
**Failing input**: `lmbda=-9.875, x=15.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 16, in <module>
    test_boxcox_inv_boxcox_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 6, in test_boxcox_inv_boxcox_roundtrip
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 13, in test_boxcox_inv_boxcox_roundtrip
    assert np.isclose(x, x_recovered, rtol=1e-6, atol=1e-6), f"Failed for lmbda={lmbda}, x={x}: recovered {x_recovered}"
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Failed for lmbda=-9.875, x=15.0: recovered 15.000022971435104
Falsifying example: test_boxcox_inv_boxcox_roundtrip(
    lmbda=-9.875,
    x=15.0,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy import special

lmbda = -10
x = 26

y = special.boxcox(x, lmbda)
x_recovered = special.inv_boxcox(y, lmbda)

print(f"Original x: {x}")
print(f"Transformed y: {y}")
print(f"Recovered x: {x_recovered}")
print(f"Relative error: {abs(x - x_recovered) / x:.6f}")
print(f"Absolute error: {abs(x - x_recovered):.6f}")

print("\nAdditional failing cases:")

# Test case 2
lmbda2 = -10
x2 = 50
y2 = special.boxcox(x2, lmbda2)
x2_recovered = special.inv_boxcox(y2, lmbda2)
print(f"\nlmbda={lmbda2}, x={x2}:")
print(f"  Recovered x: {x2_recovered}")
print(f"  Is inf: {np.isinf(x2_recovered)}")

# Test case 3
lmbda3 = -10
x3 = 100
y3 = special.boxcox(x3, lmbda3)
x3_recovered = special.inv_boxcox(y3, lmbda3)
print(f"\nlmbda={lmbda3}, x={x3}:")
print(f"  Recovered x: {x3_recovered}")
print(f"  Is inf: {np.isinf(x3_recovered)}")

# Test case 4
lmbda4 = -8
x4 = 100
y4 = special.boxcox(x4, lmbda4)
x4_recovered = special.inv_boxcox(y4, lmbda4)
print(f"\nlmbda={lmbda4}, x={x4}:")
print(f"  Recovered x: {x4_recovered}")
if not np.isinf(x4_recovered):
    print(f"  Relative error: {abs(x4 - x4_recovered) / x4:.6f}")
```

<details>

<summary>
Silent infinity returns and numerical errors demonstrated
</summary>
```
Original x: 26
Transformed y: 0.09999999999999928
Recovered x: 25.992076683399546
Relative error: 0.000305
Absolute error: 0.007923

Additional failing cases:

lmbda=-10, x=50:
  Recovered x: inf
  Is inf: True

lmbda=-10, x=100:
  Recovered x: inf
  Is inf: True

lmbda=-8, x=100:
  Recovered x: 98.70149282610821
  Relative error: 0.012985
```
</details>

## Why This Is A Bug

This bug violates fundamental mathematical properties and user expectations in several critical ways:

1. **Silent infinity returns without warnings**: When `lmbda=-10` and `x=50` or `x=100`, `inv_boxcox` silently returns infinity instead of the correct finite values. This is a catastrophic failure that could corrupt downstream calculations without any indication of error.

2. **Violates mathematical invertibility**: The Box-Cox transformation is mathematically defined as invertible for all x > 0 and finite lambda. The formula is:
   - Forward: `y = (x^λ - 1)/λ` for λ ≠ 0
   - Inverse: `x = (λ*y + 1)^(1/λ)` for λ ≠ 0

   These should be exact inverses, but the implementation fails this fundamental property.

3. **Documentation contradicts behavior**: The `inv_boxcox` docstring includes an example demonstrating successful round-trip operations without any warnings about parameter limitations or numerical stability issues. Users reasonably expect this to work for all valid inputs.

4. **Significant errors even when not returning infinity**: With `lmbda=-8, x=100`, the function returns 98.7 instead of 100, a relative error of 1.3%. This level of error is unacceptable for numerical computations that may be chained together.

5. **No parameter validation or warnings**: The functions accept parameter combinations known to cause numerical instability without any warnings, errors, or documentation of these limitations.

## Relevant Context

The root cause is numerical instability in the Box-Cox formula when lambda is very negative. For the forward transformation `y = (x^λ - 1)/λ`:
- When λ is very negative (e.g., -10) and x is large (e.g., 50), `x^λ` becomes extremely small (approaching 0)
- This causes `x^λ - 1` to approach -1, leading to catastrophic cancellation
- The transformed value y becomes close to 0.1 (= -1/-10)
- During inversion, small numerical errors in y get exponentially amplified

The scipy documentation for these functions can be found at:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.inv_boxcox.html

The implementation likely uses straightforward formulas without special handling for extreme parameter values, leading to these numerical issues.

## Proposed Fix

The bug requires implementation of numerically stable algorithms for extreme parameter values. Here's a high-level approach:

1. **Add parameter validation and warnings** to alert users when operating in numerically unstable regions
2. **Implement alternative formulations** for extreme negative lambda values using logarithmic arithmetic
3. **Never return infinity silently** - either raise an exception or use a more stable algorithm

A partial fix adding parameter validation:

```diff
def boxcox(x, lmbda):
+    # Warn about numerical instability for extreme parameters
+    if lmbda < -6 and x > 10:
+        import warnings
+        warnings.warn(
+            f"Box-Cox transformation may be numerically unstable for lambda={lmbda} and x={x}. "
+            "Consider using less extreme parameter values.",
+            RuntimeWarning
+        )

    if lmbda == 0:
        return np.log(x)
    else:
        return (x**lmbda - 1) / lmbda

def inv_boxcox(y, lmbda):
+    # Check for potential overflow before computation
+    if lmbda < 0:
+        max_safe_y = (np.finfo(float).max ** lmbda - 1) / lmbda
+        if y > max_safe_y * 0.9:  # 90% safety margin
+            raise ValueError(
+                f"inv_boxcox would overflow for y={y}, lambda={lmbda}. "
+                f"Maximum safe y value is approximately {max_safe_y:.6f}"
+            )

    if lmbda == 0:
        return np.exp(y)
    else:
-        return (lmbda * y + 1) ** (1 / lmbda)
+        result = (lmbda * y + 1) ** (1 / lmbda)
+        if np.isinf(result):
+            raise OverflowError(
+                f"inv_boxcox computation overflowed for y={y}, lambda={lmbda}"
+            )
+        return result
```

For a complete fix, the implementation should use logarithmic arithmetic for extreme negative lambda values to maintain numerical stability throughout the valid parameter range.