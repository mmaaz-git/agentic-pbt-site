# Bug Report: scipy.odr.unilinear IndexError with 1-parameter beta0

**Target**: `scipy.odr.unilinear`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.odr.unilinear` model has a misleading name and raises an `IndexError` when used with a single-parameter `beta0`, despite "unilinear" typically meaning a line through the origin (one parameter: `y = B[0] * x`).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.odr as odr

@given(n=st.integers(min_value=5, max_value=30))
def test_unilinear_model(n):
    """Property: Built-in unilinear model should work with single parameter"""
    x = np.linspace(1, 10, n)
    y = 2.5 * x

    unilin_model = odr.unilinear
    data = odr.Data(x, y)

    odr_obj = odr.ODR(data, unilin_model, beta0=[1.0])
    output = odr_obj.run()

    assert len(output.beta) == 1
    assert np.isclose(output.beta[0], 2.5, rtol=1e-5)
```

**Failing input**: `n=5`

## Reproducing the Bug

```python
import numpy as np
import scipy.odr as odr

x = np.array([1, 2, 3, 4, 5])
y = 2.5 * x

unilin_model = odr.unilinear
data = odr.Data(x, y)

odr_obj = odr.ODR(data, unilin_model, beta0=[1.0])
```

**Error:**
```
IndexError: index 1 is out of bounds for axis 0 with size 1
```

**Traceback:**
```
File "scipy/odr/_odrpack.py", line 850, in _check
    res = self.model.fcn(*arglist)
File "scipy/odr/_models.py", line 215, in _unilin
    return x*B[0] + B[1]
                    ~^^^
```

## Why This Is A Bug

The model is named `unilinear`, which in standard mathematical terminology means a linear relationship with one parameter (a line through the origin: `y = B[0] * x`). However, the implementation requires two parameters:

```python
def _unilin(B, x):
    return x*B[0] + B[1]
```

The model's metadata even confirms it expects two parameters:
```python
'equ': 'y = B_0 * x + B_1'
```

**Contract violation:** The name `unilinear` creates the reasonable expectation of a single-parameter model, but the implementation requires two parameters. This violates the principle of least astonishment and makes the API confusing.

**Workaround:** Users must provide `beta0=[slope, 0.0]`, which works but defeats the purpose of having a "unilinear" model distinct from the regular `linear` model.

## Fix

**Option 1:** Fix the implementation to match the name (recommended):

```diff
--- a/scipy/odr/_models.py
+++ b/scipy/odr/_models.py
@@ -212,7 +212,7 @@ def _unilin_est(data):
     return (y/x).mean()

 def _unilin(B, x):
-    return x*B[0] + B[1]
+    return x*B[0]

 def _unilin_fjd(B, x):
-    return np.ones(x.shape[0], float)
+    return B[0] * np.ones(x.shape[0], float)

 def _unilin_fjb(B, x):
-    return np.concatenate((x.reshape(-1, 1), np.ones((x.shape[0], 1), float)), axis=1)
+    return x.reshape(-1, 1)

 def _unilin_est(data):
-    return np.array([data.y.mean()/data.x.mean(), 0.])
+    return np.array([data.y.mean()/data.x.mean()])
```

**Option 2:** Rename the model to `linear` and deprecate `unilinear`, or clarify in documentation that `unilinear` actually has two parameters (which would be confusing).