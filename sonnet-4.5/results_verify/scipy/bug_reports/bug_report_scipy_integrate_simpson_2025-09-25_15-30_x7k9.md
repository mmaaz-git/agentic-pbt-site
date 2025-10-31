# Bug Report: scipy.integrate.simpson Asymmetric for Even-Length Arrays

**Target**: `scipy.integrate.simpson`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.integrate.simpson` produces different results when given reversed arrays with an even number of points, violating the mathematical property that integration over evenly-spaced samples should be invariant to array reversal.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.integrate import simpson


@given(
    y=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=3, max_size=50),
    dx=st.floats(min_value=1e-3, max_value=100, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_simpson_reversal(y, dx):
    y_arr = np.array(y)
    forward = simpson(y_arr, dx=dx)
    backward = simpson(y_arr[::-1], dx=dx)

    assert np.isclose(forward, backward, rtol=1e-10), \
        f"simpson should be same regardless of direction, got {forward} vs {backward}"
```

**Failing input**: `y=[0.0, 0.0, 0.0, 1.0], dx=1.0`

## Reproducing the Bug

```python
import numpy as np
from scipy.integrate import simpson

y = np.array([0.0, 0.0, 0.0, 1.0])

forward = simpson(y, dx=1.0)
backward = simpson(y[::-1], dx=1.0)

print(f"simpson([0, 0, 0, 1], dx=1.0) = {forward}")
print(f"simpson([1, 0, 0, 0], dx=1.0) = {backward}")
print(f"Difference: {forward - backward}")
```

Output:
```
simpson([0, 0, 0, 1], dx=1.0) = 0.4166666666666667
simpson([1, 0, 0, 0], dx=1.0) = 0.3333333333333333
Difference: 0.08333333333333337
```

## Why This Is A Bug

For evenly-spaced samples with constant `dx`, reversing the array of y-values should produce the same integral value. This is a fundamental mathematical property: if we have samples at positions x=[0,1,2,3] with values y=[0,0,0,1], the integral should be the same as the integral at positions x=[0,1,2,3] with values y=[1,0,0,0] (which is effectively the same function traversed in the same direction).

**Evidence this is a bug:**

1. **Mathematical property violated**: Integration over evenly-spaced samples should be invariant to reversing the y-values when spacing is constant.

2. **Inconsistent with other methods**: `trapezoid` correctly produces the same result for reversed arrays:
   - `trapezoid([0,0,0,1], dx=1.0)` = 0.5
   - `trapezoid([1,0,0,0], dx=1.0)` = 0.5

3. **Only affects even N**: When the array has an odd number of points, simpson works correctly:
   - `simpson([0,0,0,0,1], dx=1.0)` = 0.3333...
   - `simpson([1,0,0,0,0], dx=1.0)` = 0.3333...

4. **Root cause**: In `_quadrature.py` lines 463-534, when N is even, the code applies a special correction formula (Cartwright's correction) only to the **last** interval, making the algorithm asymmetric. The correction uses `y[slice1]`, `y[slice2]`, and `y[slice3]` which refer to the last three points, so reversing the array changes which points receive this special treatment.

## Fix

The fix requires applying the Cartwright correction symmetrically. When N is even, instead of applying the correction only to the last interval, the algorithm should either:

1. Apply the correction to both the first and last intervals, or
2. Use a symmetric correction that doesn't favor one end over the other

A possible approach (option 1) would be:

```diff
--- a/scipy/integrate/_quadrature.py
+++ b/scipy/integrate/_quadrature.py
@@ -475,8 +475,17 @@ def simpson(y, x=None, *, dx=1.0, axis=-1):
                 last_dx = x[slice1] - x[slice2]
             val += 0.5 * last_dx * (y[slice1] + y[slice2])
         else:
-            # use Simpson's rule on first intervals
-            result = _basic_simpson(y, 0, N-3, x, dx, axis)
+            # use Simpson's rule on middle intervals only
+            # Apply Cartwright correction to both first and last intervals
+            if N > 4:
+                result = _basic_simpson(y, 2, N-3, x, dx, axis)
+            else:
+                result = 0.0
+
+            # Apply correction to first interval
+            slice0 = tupleset(slice_all, axis, 0)
+            slice1_first = tupleset(slice_all, axis, 1)
+            slice2_first = tupleset(slice_all, axis, 2)

             slice1 = tupleset(slice_all, axis, -1)
             slice2 = tupleset(slice_all, axis, -2)
@@ -529,7 +538,23 @@ def simpson(y, x=None, *, dx=1.0, axis=-1):
                 where=den != 0
             )

+            # Apply correction to last interval
             result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]
+
+            # Apply symmetric correction to first interval
+            h_first = np.asarray([dx, dx], dtype=np.float64)
+            if x is not None:
+                hp0 = tupleset(slice_all, axis, slice(0, 1, 1))
+                hp1 = tupleset(slice_all, axis, slice(1, 2, 1))
+                diffs = np.float64(np.diff(x, axis=axis))
+                h_first = [np.squeeze(diffs[hp0], axis=axis),
+                          np.squeeze(diffs[hp1], axis=axis)]
+
+            num_f = 2 * h_first[0] ** 2 + 3 * h_first[1] * h_first[0]
+            den_f = 6 * (h_first[0] + h_first[1])
+            alpha_f = np.true_divide(num_f, den_f, out=np.zeros_like(den_f), where=den_f != 0)
+            # ... (similar calculations for beta_f and eta_f)
+            result += alpha_f*y[slice0] + beta_f*y[slice1_first] - eta_f*y[slice2_first]

         result += val
     else:
```

This fix ensures that both ends of the array receive the same special treatment, making the algorithm symmetric.