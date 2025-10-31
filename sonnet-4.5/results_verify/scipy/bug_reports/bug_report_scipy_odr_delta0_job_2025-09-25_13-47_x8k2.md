# Bug Report: scipy.odr TypeError when using delta0 without specifying job

**Target**: `scipy.odr.ODR.run()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Using `delta0` parameter in ODR without explicitly specifying `job` causes a `TypeError` when calling `run()`, due to attempting arithmetic operations on `None`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.odr as odr

@given(n=st.integers(min_value=5, max_value=30))
def test_delta0_initialization(n):
    """Property: delta0 can be provided for initialization"""
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 + np.random.RandomState(42).randn(n) * 0.1

    def linear_func(B, x):
        return B[0] * x + B[1]

    model = odr.Model(linear_func)
    data = odr.Data(x, y)

    delta0 = np.zeros(n)

    odr_obj = odr.ODR(data, model, beta0=[1.0, 0.0], delta0=delta0)
    output = odr_obj.run()

    assert hasattr(output, 'delta')
```

**Failing input**: `n=5`

## Reproducing the Bug

```python
import numpy as np
import scipy.odr as odr

n = 5
x = np.linspace(0, 10, n)
y = 2 * x + 1

def linear_func(B, x):
    return B[0] * x + B[1]

model = odr.Model(linear_func)
data = odr.Data(x, y)
delta0 = np.zeros(n)

odr_obj = odr.ODR(data, model, beta0=[1.0, 0.0], delta0=delta0)
output = odr_obj.run()
```

**Error:**
```
TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'
```

**Traceback:**
```
File "scipy/odr/_odrpack.py", line 1100, in run
    if self.delta0 is not None and (self.job // 10000) % 10 == 0:
                                    ^^^^^^^^
TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'
```

## Why This Is A Bug

The code in `ODR.run()` attempts arithmetic operations on `self.job` without first checking if it's `None`. When the user doesn't specify the `job` parameter (which is an advanced parameter most users won't know about), it defaults to `None`, causing the crash.

This is a crash on valid inputs - `delta0` is a documented parameter that users should be able to use without needing to understand the internal `job` parameter.

## Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1097,7 +1097,7 @@ class ODR:

     def run(self):
         ...
-        if self.delta0 is not None and (self.job // 10000) % 10 == 0:
+        if self.delta0 is not None and self.job is not None and (self.job // 10000) % 10 == 0:
             ...
```

Alternatively, ensure `self.job` has a sensible default integer value instead of `None` during initialization.