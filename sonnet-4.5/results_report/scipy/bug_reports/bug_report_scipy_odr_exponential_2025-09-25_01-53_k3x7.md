# Bug Report: scipy.odr.exponential Converges to Wrong Parameters

**Target**: `scipy.odr.exponential`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `scipy.odr.exponential` model converges to incorrect parameters when fitting exact exponential data without user-provided initial parameter estimates, due to a poor default estimation function that always returns `[1.0, 1.0]` regardless of the data.

## Property-Based Test

```python
import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from scipy import odr


@given(
    beta0=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    beta1=st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_exponential_model_fitting_property(beta0, beta1):
    x = np.linspace(0.0, 5.0, 20)
    y = beta0 + np.exp(beta1 * x)

    assume(np.all(np.isfinite(y)))
    assume(np.max(np.abs(y)) < 1e10)

    data = odr.Data(x, y)

    odr_obj_with_init = odr.ODR(data, odr.exponential, beta0=[beta0, beta1])
    output_with_init = odr_obj_with_init.run()

    y_fitted_with_init = output_with_init.beta[0] + np.exp(output_with_init.beta[1] * x)
    residuals_with_init = y - y_fitted_with_init
    ssr_with_init = np.sum(residuals_with_init**2)

    assert ssr_with_init < 1e-10

    odr_obj_without_init = odr.ODR(data, odr.exponential)
    output_without_init = odr_obj_without_init.run()

    y_fitted_without_init = output_without_init.beta[0] + np.exp(output_without_init.beta[1] * x)
    residuals_without_init = y - y_fitted_without_init
    ssr_without_init = np.sum(residuals_without_init**2)

    assert ssr_without_init < 1e-10
```

**Failing input**: `beta0=0.0, beta1=2.0`

## Reproducing the Bug

```python
import numpy as np
from scipy import odr

x = np.linspace(0.0, 5.0, 20)
y_true = 0.0 + np.exp(2.0 * x)

data = odr.Data(x, y_true)
odr_obj = odr.ODR(data, odr.exponential)
output = odr_obj.run()

print(f"True parameters: beta0=0.0, beta1=2.0")
print(f"Recovered: beta0={output.beta[0]}, beta1={output.beta[1]}")

y_fitted = output.beta[0] + np.exp(output.beta[1] * x)
residuals = y_true - y_fitted
ssr = np.sum(residuals**2)

print(f"Sum of squared residuals: {ssr}")
```

**Output:**
```
True parameters: beta0=0.0, beta1=2.0
Recovered: beta0=0.81632061527811, beta1=1.9083302467939744
Sum of squared residuals: 96329616.4465015
```

**Expected:**
```
True parameters: beta0=0.0, beta1=2.0
Recovered: beta0=0.0, beta1=2.0
Sum of squared residuals: ~0.0
```

## Why This Is A Bug

1. The exponential model documentation shows that it should fit exact exponential data correctly (see docstring example).
2. When correct initial parameters are provided (`beta0=[0.0, 2.0]`), ODR achieves zero residuals, proving the optimization algorithm *can* find the correct solution.
3. Without initial parameters, ODR converges to a local minimum with **96 million** sum of squared residuals instead of ~0.
4. The root cause is in `_exp_est()` in `scipy/odr/_models.py`, which returns `[1., 1.]` for all data (note the comment "# Eh." indicating this is a known placeholder).

This affects real users because:
- Most users won't provide initial parameter estimates
- The fitted model will have massive errors
- The convergence appears successful (no exceptions), silently returning wrong results
- Affects a wide range of parameter values (not just edge cases)

## Fix

The issue is in `scipy/odr/_models.py`:

```diff
def _exp_est(data):
-    # Eh.
-    return np.array([1., 1.])
+    # Estimate parameters for y = beta0 + exp(beta1 * x)
+    # Use a simple heuristic: estimate beta1 from slope of log(y - y_min)
+    x = data.x
+    y = data.y
+
+    # Estimate beta0 as minimum y value (since exp() >= 0)
+    beta0_est = np.min(y) - 1.0
+
+    # Shift y to ensure positivity for log
+    y_shifted = y - beta0_est
+    y_shifted = np.maximum(y_shifted, 1e-10)
+
+    # Estimate beta1 from linear regression on log(y_shifted)
+    log_y = np.log(y_shifted)
+
+    if x.ndim == 1:
+        # Simple linear regression: beta1 = cov(x, log_y) / var(x)
+        beta1_est = np.cov(x, log_y)[0, 1] / np.var(x)
+    else:
+        # For multidimensional x, use first dimension
+        beta1_est = np.cov(x[0], log_y)[0, 1] / np.var(x[0])
+
+    return np.array([beta0_est, beta1_est])
```

This improved estimator:
1. Estimates `beta0` from the minimum y value (since exp() is always positive)
2. Estimates `beta1` by taking logs and doing linear regression
3. Provides much better initial parameters for the optimizer
4. Handles edge cases gracefully