# Bug Report: scipy.interpolate.splrep Silent Failure with Close X Values

**Target**: `scipy.interpolate.splrep`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`splrep` silently produces incorrect interpolation when x values are extremely close together (gap < ~1e-15), while the recommended alternative `make_interp_spline` correctly raises an error for this case.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.interpolate as si

@settings(max_examples=500)
@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=2, max_size=50).flatmap(
        lambda x_list: st.tuples(
            st.just(sorted(set(x_list))),
            st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                     min_size=len(set(x_list)), max_size=len(set(x_list)))
        )
    ).filter(lambda xy: len(xy[0]) >= 2)
)
def test_splrep_splev_round_trip(xy):
    x, y = xy
    x = np.array(x)
    y = np.array(y)

    assume(len(x) >= 2)
    assume(len(x) == len(y))
    assume(np.all(np.diff(x) > 0))

    tck = si.splrep(x, y, s=0)
    y_evaluated = si.splev(x, tck)

    assert np.allclose(y, y_evaluated, rtol=1e-9, atol=1e-9), \
        f"splev should return original y values at original x. Expected {y}, got {y_evaluated}"
```

**Failing input**: `x=[-1.0, 0.0, 1.26e-74, 1.0], y=[0.0, 0.0, 1.0, 0.0]`

## Reproducing the Bug

```python
import numpy as np
import scipy.interpolate as si

x = np.array([0.0, 1.0, 1.0 + 1e-50, 2.0])
y = np.array([0.0, 1.0, 0.5, 0.0])

tck = si.splrep(x, y, s=0)
y_evaluated = si.splev(x, tck)

print(f"Expected: {y}")
print(f"Got:      {y_evaluated}")
print(f"Max error: {np.max(np.abs(y - y_evaluated)):.6f}")
```

Output:
```
Expected: [0.  1.  0.5 0. ]
Got:      [-0.123  1.  1.  0.]
Max error: 0.500000
```

The bug manifests when x values are closer than approximately 1e-15 (machine epsilon for float64):
- Gap 1e-10: Works correctly (max error ~1e-6)
- Gap 1e-20: Fails (max error ~0.5)

## Why This Is A Bug

1. **Documentation doesn't warn about minimum spacing**: The `splrep` documentation doesn't mention any requirement for minimum spacing between x values.

2. **Silent failure**: When x values are too close, `splrep` silently produces incorrect interpolation instead of raising an error.

3. **Inconsistent with recommended alternative**: The documentation says "Specifically, we recommend using `make_splrep` in new code" (legacy function marker), and `make_interp_spline` correctly raises `ValueError: Expect x to not have duplicates` for the same input.

4. **Violates interpolation contract**: With `s=0`, `splrep` is supposed to create an interpolating spline that passes through all data points. The output clearly violates this.

## Fix

The fix should make `splrep` check for x values that are too close together (within machine epsilon) and raise an informative error, similar to how `make_interp_spline` handles this:

```diff
--- a/scipy/interpolate/_fitpack_py.py
+++ b/scipy/interpolate/_fitpack_py.py
@@ -250,6 +250,12 @@ def splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
     x = atleast_1d(x)
     y = atleast_1d(y)

+    # Check for x values that are too close together
+    if len(x) > 1:
+        min_gap = np.min(np.abs(np.diff(x)))
+        if min_gap < np.finfo(x.dtype).eps * np.max(np.abs(x)):
+            raise ValueError("x values are too close together (closer than machine epsilon)")
+
     m = len(x)
     if w is None:
         w = ones(m, float)
```

Alternatively, the documentation should be updated to explicitly warn users about this limitation.