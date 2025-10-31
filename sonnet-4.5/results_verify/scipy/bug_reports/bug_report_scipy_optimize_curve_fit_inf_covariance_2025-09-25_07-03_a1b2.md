# Bug Report: scipy.optimize.curve_fit returns infinite covariance matrix

**Target**: `scipy.optimize.curve_fit`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.optimize.curve_fit` returns a covariance matrix filled with `inf` values when fitting exact data with certain parameter values (specifically when the intercept is 0), despite issuing a warning that "Covariance of the parameters could not be estimated". This creates an inconsistent API where the function warns about a problem but still returns invalid data instead of raising an exception or returning None.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from scipy import optimize
import numpy as np

@settings(max_examples=500)
@given(
    scale=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
    offset=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
)
def test_curve_fit_covariance_shape(scale, offset):
    def model(x, a, b):
        return a * x + b

    xdata = np.linspace(-10, 10, 50)
    ydata = model(xdata, scale, offset)

    try:
        popt, pcov = optimize.curve_fit(model, xdata, ydata)

        assert popt.shape == (2,), f"popt shape {popt.shape}, expected (2,)"
        assert pcov.shape == (2, 2), f"pcov shape {pcov.shape}, expected (2, 2)"
        assert np.all(np.isfinite(popt)), "popt contains non-finite values"
        assert np.all(np.isfinite(pcov)), "pcov contains non-finite values"
    except (ValueError, RuntimeError):
        pass
```

**Failing input**: `scale=1.015625, offset=0.0`

## Reproducing the Bug

```python
import numpy as np
from scipy import optimize

def model(x, a, b):
    return a * x + b

scale = 1.015625
offset = 0.0

xdata = np.linspace(-10, 10, 50)
ydata = model(xdata, scale, offset)

popt, pcov = optimize.curve_fit(model, xdata, ydata)

print(f"Parameters recovered: {popt}")
print(f"Covariance matrix:\n{pcov}")
print(f"Contains inf: {np.any(np.isinf(pcov))}")
```

Output:
```
Parameters recovered: [1.015625 0.]
Covariance matrix:
[[inf inf]
 [inf inf]]
Contains inf: True
```

## Why This Is A Bug

The function issues a warning "Covariance of the parameters could not be estimated" but then proceeds to return a covariance matrix filled with `inf` values. This violates the API contract in several ways:

1. **Inconsistent error handling**: If the covariance cannot be estimated (as the warning states), the function should either:
   - Raise an exception (preferred for failures)
   - Return None or NaN for the covariance matrix
   - Document that `inf` values indicate invalid covariance

2. **No programmatic way to check validity**: Users must check for `inf` values themselves rather than catching an exception, making error handling awkward.

3. **Unexpected behavior for valid input**: The input data is perfectly valid (fitting a linear model to exact data points). The function successfully recovers the parameters (`popt` is correct), so the fit succeeded, but the covariance estimation fails.

4. **Happens with realistic parameter values**: This occurs when the intercept is exactly 0 and the x-data is symmetric around 0, which is not an unusual edge case.

## Fix

The issue occurs because when the intercept (`b`) is exactly 0 and `xdata` is symmetric around 0, the fitted residuals are exactly zero (perfect fit), causing the covariance estimation formula to divide by zero.

The fix should detect when covariance cannot be computed and either:

**Option 1** (Preferred): Raise an informative exception
```diff
--- a/scipy/optimize/_minpack_py.py
+++ b/scipy/optimize/_minpack_py.py
@@ -493,7 +493,10 @@ def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
     if pcov is None:
         # indeterminate covariance
         pcov = zeros((len(popt), len(popt)), dtype=float)
         pcov.fill(inf)
-        warn('Covariance of the parameters could not be estimated',
+        raise ValueError(
+            'Covariance of the parameters could not be estimated. '
+            'This can occur when fitting exact data or when the model '
+            'is poorly conditioned. Consider adding regularization or noise.',
              OptimizeWarning)
```

**Option 2**: Return NaN instead of inf
```diff
--- a/scipy/optimize/_minpack_py.py
+++ b/scipy/optimize/_minpack_py.py
@@ -493,7 +493,7 @@ def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
     if pcov is None:
         # indeterminate covariance
         pcov = zeros((len(popt), len(popt)), dtype=float)
-        pcov.fill(inf)
+        pcov.fill(nan)
         warn('Covariance of the parameters could not be estimated',
              OptimizeWarning)
```

Option 1 is preferred because it makes the error explicit and forces users to handle the case, rather than silently returning invalid data.