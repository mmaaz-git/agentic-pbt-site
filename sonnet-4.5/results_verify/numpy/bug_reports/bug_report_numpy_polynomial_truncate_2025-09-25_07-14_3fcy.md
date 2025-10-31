# Bug Report: numpy.polynomial truncate() Doesn't Match Documented Behavior

**Target**: `numpy.polynomial.polynomial.Polynomial.truncate`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate(size)` method's docstring states it will "Truncate series to length `size`", but when `size` is greater than the current coefficient array length, it returns a polynomial with length less than `size`, violating the documented contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.polynomial as np_poly

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=20),
    st.integers(min_value=1, max_value=20)
)
@settings(max_examples=500)
def test_truncate_size(coefs, size):
    p = np_poly.Polynomial(coefs)
    p_trunc = p.truncate(size)
    assert len(p_trunc.coef) == size
```

**Failing input**: `coefs=[0.0], size=2`

## Reproducing the Bug

```python
import numpy.polynomial as np_poly

p = np_poly.Polynomial([1])
p_trunc = p.truncate(3)
print(f"Length: {len(p_trunc.coef)}, Expected: 3")

p2 = np_poly.Polynomial([0.0])
p2_trunc = p2.truncate(2)
print(f"Length: {len(p2_trunc.coef)}, Expected: 2")
```

Output:
```
Length: 1, Expected: 3
Length: 1, Expected: 2
```

## Why This Is A Bug

The docstring explicitly states: "Truncate series to length `size`" and "Reduce the series to length `size`". This clearly promises that the returned series will have exactly `size` coefficients. However, when `size > len(coef)`, the method returns the original polynomial unchanged, resulting in `len(result) < size`.

For comparison, the `cutdeg` method explicitly documents what happens when the requested degree exceeds the current degree: "If `deg` is greater than the current degree a copy of the current series is returned." The `truncate` method lacks this clarification, creating a misleading contract.

## Fix

The fix should clarify the documentation to match the actual behavior:

```diff
--- a/numpy/polynomial/_polybase.py
+++ b/numpy/polynomial/_polybase.py
@@ -760,8 +760,9 @@ class ABCPolyBase(abc.ABC):
     def truncate(self, size):
-        """Truncate series to length `size`.
+        """Truncate series to at most length `size`.

-        Reduce the series to length `size` by discarding the high
+        Reduce the series to at most length `size` by discarding the high
         degree terms. The value of `size` must be a positive integer. This
+        If `size` is greater than the current length, a copy of the current
+        series is returned.
         can be useful in least squares where the coefficients of the
```