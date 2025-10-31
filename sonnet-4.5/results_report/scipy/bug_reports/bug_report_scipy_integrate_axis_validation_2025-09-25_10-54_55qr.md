# Bug Report: scipy.integrate Invalid Axis Error Messages

**Target**: `scipy.integrate.trapezoid`, `scipy.integrate.simpson`, `scipy.integrate.cumulative_trapezoid`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When an invalid `axis` parameter is provided to `trapezoid`, `simpson`, or `cumulative_trapezoid`, these functions raise unclear `IndexError` messages instead of informative error messages like `ValueError` or `AxisError`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy import integrate
import pytest


@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_trapezoid_invalid_axis_error_message(dim1, dim2):
    y = np.ones((dim1, dim2))
    invalid_axis = y.ndim + 1

    with pytest.raises((ValueError, Exception)) as exc_info:
        integrate.trapezoid(y, axis=invalid_axis)

    assert 'axis' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower(), \
        f"Error message should mention 'axis' or 'bound', got: {exc_info.value}"
```

**Failing input**: Any 2D array with `axis >= ndim`

## Reproducing the Bug

```python
import numpy as np
from scipy import integrate

y = np.array([[1, 2, 3],
              [4, 5, 6]])

integrate.trapezoid(y, axis=2)

integrate.simpson(y, axis=2)

integrate.cumulative_trapezoid(y, axis=2)
```

**Current output:**
```
IndexError: list assignment index out of range              # trapezoid
IndexError: tuple index out of range                        # simpson
IndexError: tuple index out of range                        # cumulative_trapezoid
```

**Expected output:**
```
ValueError: `axis=2` is not valid for `y` with `y.ndim=2`.  # like cumulative_simpson
AxisError: axis 2 is out of bounds for array of dimension 2  # like numpy.sum
```

## Why This Is A Bug

The error messages are unclear and do not indicate what the actual problem is. Users cannot tell from "list assignment index out of range" that they provided an invalid axis parameter. This violates the principle of least surprise and makes debugging difficult.

Notably, `scipy.integrate.cumulative_simpson` already validates axis correctly with a clear error message, showing that the library knows how to do this properly. The inconsistency makes the bug more egregious.

## Fix

Add axis validation at the beginning of each function. Following the pattern from `cumulative_simpson`:

```diff
diff --git a/scipy/integrate/_quadrature.py b/scipy/integrate/_quadrature.py
index 1234567..abcdefg 100644
--- a/scipy/integrate/_quadrature.py
+++ b/scipy/integrate/_quadrature.py
@@ -100,6 +100,10 @@ def trapezoid(y, x=None, dx=1.0, axis=-1):
     """
     y = np.asanyarray(y)

+    if axis >= y.ndim or axis < -y.ndim:
+        raise ValueError(
+            f"`axis={axis}` is not valid for `y` with `y.ndim={y.ndim}`."
+        )
+
     if x is None:
         d = dx
     else:
```

Similar validation should be added to `simpson` and `cumulative_trapezoid`.