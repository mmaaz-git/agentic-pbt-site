# Bug Report: numpy.polynomial.polyint crashes on empty input

**Target**: `numpy.polynomial.polynomial.polyint`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`polyint` crashes with `IndexError` when given an empty coefficient array, while the related function `polyder` handles empty input gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@given(st.lists(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
                min_size=0, max_size=8))
@settings(max_examples=500)
def test_polyint_polyder_consistency(c):
    """Test that polyint and polyder handle inputs consistently"""
    try:
        der_result = poly.polyder(c)
        int_result = poly.polyint(c)
    except Exception as e:
        assert type(e).__name__ == 'ValueError', \
            f"Should raise ValueError for invalid input, not {type(e).__name__}"
```

**Failing input**: `[]` (empty list)

## Reproducing the Bug

```python
import numpy.polynomial.polynomial as poly

result_der = poly.polyder([])
print(f"polyder([]) = {result_der}")

result_int = poly.polyint([])
```

**Output**:
```
polyder([]) = []
Traceback (most recent call last):
  File "reproduce.py", line 6, in <module>
    result_int = poly.polyint([])
  File ".../numpy/polynomial/polynomial.py", line 653, in polyint
    tmp[0] = c[0] * 0
             ~^^^
IndexError: index 0 is out of bounds for axis 0 with size 0
```

## Why This Is A Bug

1. **Inconsistent error handling**: `polyder` gracefully handles empty input by returning an empty array, but `polyint` crashes with an `IndexError`.

2. **Poor error message**: If empty input is invalid, the function should raise a `ValueError` with a clear message, not an `IndexError`.

3. **Violates symmetry**: As inverse operations, `polyder` and `polyint` should handle edge cases consistently.

## Fix

Add empty input handling to `polyint` similar to `polyder`:

```diff
def polyint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c + 0.0
    cdt = c.dtype
    if not np.iterable(k):
        k = [k]
    cnt = pu._as_int(m, "the order of integration")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    k = list(k) + [0] * (cnt - len(k))
    c = np.moveaxis(c, iaxis, 0)
+   n = len(c)
+   if n == 0:
+       return c
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=cdt)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            tmp[0] += k[i] - polyval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c
```