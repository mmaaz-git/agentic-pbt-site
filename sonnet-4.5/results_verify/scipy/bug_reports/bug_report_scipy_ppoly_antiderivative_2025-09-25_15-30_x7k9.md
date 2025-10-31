# Bug Report: scipy.interpolate.PPoly antiderivative loses piecewise structure

**Target**: `scipy.interpolate.PPoly.antiderivative`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When taking the antiderivative of a derivative of a discontinuous piecewise polynomial (e.g., piecewise constant), `PPoly` loses the original piecewise structure. The docstring claims "derivative is its inverse operation" for antiderivative, but `p.derivative().antiderivative()` does not equal `p` up to a constant for discontinuous piecewise functions.

## Property-Based Test

```python
import numpy as np
from scipy.interpolate import PPoly
from hypothesis import given, strategies as st, settings, assume

@given(
    n_intervals=st.integers(min_value=1, max_value=10),
    k=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=1000000)
)
@settings(max_examples=500)
def test_ppoly_derivative_antiderivative_inverse(n_intervals, k, seed):
    rng = np.random.RandomState(seed)

    x = np.sort(rng.uniform(-10, 10, n_intervals + 1))
    assume(len(np.unique(x)) == n_intervals + 1)
    assume(np.all(np.diff(x) > 1e-6))

    c = rng.uniform(-10, 10, (k + 1, n_intervals))

    p = PPoly(c, x)
    p_deriv = p.derivative()
    p_deriv_anti = p_deriv.antiderivative()

    x_test = np.linspace(x[0], x[-1], 50)
    diff = p(x_test) - p_deriv_anti(x_test)

    is_constant = np.allclose(diff - diff[0], 0, atol=1e-8)
    assert is_constant, \
        f"derivative().antiderivative() should differ from original by constant"
```

**Failing input**: `n_intervals=2, k=0` (piecewise constant with 2 intervals)

## Reproducing the Bug

```python
import numpy as np
from scipy.interpolate import PPoly

c = np.array([[1.0, 2.0]])
x = np.array([0.0, 1.0, 2.0])
p = PPoly(c, x)

p_deriv = p.derivative()
p_deriv_anti = p_deriv.antiderivative()

print(f"p(0.5) = {p(0.5)}, p(1.5) = {p(1.5)}")
print(f"p'_anti(0.5) = {p_deriv_anti(0.5)}, p'_anti(1.5) = {p_deriv_anti(1.5)}")

diff_at_05 = p(0.5) - p_deriv_anti(0.5)
diff_at_15 = p(1.5) - p_deriv_anti(1.5)
print(f"\nDifference at 0.5: {diff_at_05}")
print(f"Difference at 1.5: {diff_at_15}")
print(f"Bug: Difference changes from {diff_at_05} to {diff_at_15} (not constant!)")
```

Output:
```
p(0.5) = 1.0, p(1.5) = 2.0
p'_anti(0.5) = 0.0, p'_anti(1.5) = 0.0

Difference at 0.5: 1.0
Difference at 1.5: 2.0
Bug: Difference changes from 1.0 to 2.0 (not constant!)
```

## Why This Is A Bug

The docstring for `PPoly.antiderivative` states: "Antiderivative is also the indefinite integral of the function, and derivative is its inverse operation."

For a piecewise constant function:
- `p(x) = 1` on [0,1] and `p(x) = 2` on [1,2]
- `p'(x) = 0` everywhere (within each piece)
- The antiderivative of `p'` should equal `p` up to a constant: `∫ p'(x) dx = p(x) + C`

However, scipy returns `antiderivative(derivative(p)) = 0` everywhere, meaning the difference `p - antiderivative(derivative(p))` is not constant across pieces (it's 1.0 in first piece, 2.0 in second piece).

This violates the claimed inverse relationship and basic calculus: the antiderivative of zero should be an arbitrary constant, not necessarily zero.

## Fix

The issue is that `antiderivative()` uses `fix_continuity()` to ensure the result is continuous, which forces the antiderivative of a zero polynomial to be 0 everywhere. For discontinuous piecewise polynomials, this loses information.

One potential fix is to document this limitation clearly, or to modify `antiderivative()` to preserve piecewise structure when the input polynomial has jump discontinuities. However, this is mathematically subtle since the derivative at discontinuities is not well-defined.

A simpler fix: update the docstring to clarify that the inverse relationship `p.derivative().antiderivative() ≈ p + C` only holds for continuous piecewise polynomials, not for discontinuous ones.

Alternatively, `antiderivative()` could set the constant term to preserve the original function's value at the start of each interval, rather than forcing continuity and setting everything to zero.