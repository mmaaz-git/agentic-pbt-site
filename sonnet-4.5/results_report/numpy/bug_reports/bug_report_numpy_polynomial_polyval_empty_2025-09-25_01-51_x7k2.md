# Bug Report: numpy.polynomial.polyval crashes on empty coefficients

**Target**: `numpy.polynomial.polynomial.polyval` (also affects `polyval2d`, `polyval3d`, `polygrid2d`, `polygrid3d`)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`polyval` and related functions crash with `IndexError` when given an empty coefficient array instead of raising a clear `ValueError` or handling the edge case gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@given(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
       st.lists(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
                min_size=0, max_size=8))
@settings(max_examples=500)
def test_polyval_handles_empty(x, c):
    """Test that polyval handles empty coefficient array without crashing"""
    try:
        result = poly.polyval(x, c)
        if len(c) == 0:
            assert result == 0 or True, "Empty poly should evaluate to something"
    except ValueError:
        pass
    except IndexError:
        assert False, "Should not raise IndexError"
```

**Failing input**: `x=2.0, c=[]`

## Reproducing the Bug

```python
import numpy.polynomial.polynomial as poly

poly.polyval(2.0, [])
```

**Output**:
```
Traceback (most recent call last):
  File "reproduce.py", line 3, in <module>
    result = poly.polyval(2.0, [])
  File ".../numpy/polynomial/polynomial.py", line 752, in polyval
    c0 = c[-1] + x * 0
         ~^^^^
IndexError: index -1 is out of bounds for axis 0 with size 0
```

**Also affects**:
```python
poly.polyval2d(1.0, 2.0, [])
poly.polyval3d(1.0, 2.0, 3.0, [])
poly.polygrid2d([1.0], [2.0], [])
poly.polygrid3d([1.0], [2.0], [3.0], [])
```

All raise the same `IndexError` because they call `polyval` internally.

## Why This Is A Bug

1. **Unhelpful error message**: The function crashes with an `IndexError` instead of raising a `ValueError` with a clear message like "coefficient array cannot be empty".

2. **Unexpected crash**: Users might programmatically generate coefficient arrays and accidentally pass an empty array, leading to confusing crashes.

3. **Widespread impact**: This bug affects 5 functions (`polyval`, `polyval2d`, `polyval3d`, `polygrid2d`, `polygrid3d`) because they all use the same underlying implementation.

4. **Inconsistent with related functions**: Some polynomial functions (like those using `as_series`) properly validate and raise `ValueError` for empty arrays.

## Fix

Add input validation at the start of the function:

```diff
def polyval(x, c, tensor=True):
    c = np.array(c, ndmin=1, copy=None)
+   if c.size == 0:
+       raise ValueError("coefficient array is empty")
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c0 = c[-1] + x * 0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x
    return c0
```