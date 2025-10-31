# Bug Report: scipy.interpolate.PPoly.roots() Returns NaN for Constant Zero Polynomial

**Target**: `scipy.interpolate.PPoly.roots()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`PPoly.roots()` returns NaN values when called on a piecewise polynomial representing a constant zero function, instead of returning an empty array or raising an informative error.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.interpolate import PPoly


@settings(max_examples=500)
@given(
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=2, max_size=10),
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=10)
)
def test_ppoly_roots_are_zeros(x_values, c_values):
    x = np.array(sorted(set(x_values)))
    assume(len(x) >= 2)

    k = len(c_values)
    c = np.array(c_values).reshape(k, 1)

    try:
        pp = PPoly(c, x)
        roots = pp.roots()

        if len(roots) > 0:
            root_values = pp(roots)
            assert np.allclose(root_values, 0, atol=1e-8), \
                f"PPoly.roots() returned {roots} but pp(roots) = {root_values}, not zeros"
    except (ValueError, np.linalg.LinAlgError):
        assume(False)
```

**Failing input**: `x_values=[0.0, 1.0], c_values=[0.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.interpolate import PPoly

x = np.array([0.0, 1.0])
c = np.array([[0.0]])

pp = PPoly(c, x)
roots = pp.roots()

print(f"Roots: {roots}")
print(f"Contains NaN: {np.any(np.isnan(roots))}")

assert not np.any(np.isnan(roots)), "PPoly.roots() should never return NaN"
```

Output:
```
Roots: [ 0. nan]
Contains NaN: True
AssertionError: PPoly.roots() should never return NaN
```

## Why This Is A Bug

1. **Violates API contract**: The documentation states that `roots()` returns "Roots of the polynomial(s)" as an ndarray. NaN is not a valid root value.

2. **Silent failure**: The method returns NaN without any warning or error, causing downstream computations to silently produce incorrect results.

3. **Unexpected behavior**: For a constant zero polynomial (where every point is technically a root), returning an array with NaN is nonsensical. Better alternatives would be:
   - Return an empty array
   - Return a special sentinel value
   - Raise an informative exception

4. **Real-world impact**: NaN values propagate through numerical computations, corrupting results and making bugs hard to debug.

## Fix

The issue occurs because `PPoly.roots()` calls `PPoly.solve(0, ...)` internally, and the root-finding logic doesn't handle the degenerate case of a constant zero polynomial correctly.

A fix should:
1. Detect when a polynomial piece is constant zero
2. Either return an empty array for that piece or handle it specially
3. Ensure no NaN values are ever returned

Without access to modify the source, the high-level fix would be in the `solve` method of `_interpolate.py` to add a check for constant polynomials and handle them appropriately.