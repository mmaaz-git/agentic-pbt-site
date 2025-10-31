# Bug Report: scipy.odr.ODR.set_iprint ValueError on Invalid Input

**Target**: `scipy.odr.ODR.set_iprint`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ODR.set_iprint()` method raises an unhelpful `ValueError` when given invalid parameter values instead of either validating inputs with a clear error message or silently ignoring invalid values like `set_job()` does.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.odr import Data, Model, ODR
import numpy as np


@given(
    init=st.integers(min_value=-5, max_value=10),
    so_init=st.integers(min_value=-5, max_value=10),
)
@settings(max_examples=200)
def test_set_iprint_invalid_values(init, so_init):
    def fcn(B, x):
        return B[0] * x + B[1]

    model = Model(fcn)
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    data = Data(x, y)
    odr_obj = ODR(data, model, beta0=[1.0, 0.0], rptfile="test.txt")

    ip2arg = [[0, 0], [1, 0], [2, 0], [1, 1], [2, 1], [1, 2], [2, 2]]

    expected_valid = [init, so_init] in ip2arg

    try:
        odr_obj.set_iprint(init=init, so_init=so_init)
        if not expected_valid:
            assert False, f"Should have raised ValueError for invalid combination [{init}, {so_init}]"
    except ValueError as e:
        if expected_valid:
            assert False, f"Should not have raised ValueError for valid combination [{init}, {so_init}]"
        if "[" not in str(e) and "not in list" not in str(e):
            assert False, f"ValueError message is unhelpful: {e}"
```

**Failing input**: `init=3, so_init=0` (or any invalid combination)

## Reproducing the Bug

```python
import numpy as np
from scipy.odr import Data, Model, ODR


def fcn(B, x):
    return B[0] * x + B[1]


model = Model(fcn)
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
odr_obj = ODR(data, model, beta0=[1.0, 0.0], rptfile="test.txt")

odr_obj.set_iprint(init=3)
```

**Output**:
```
ValueError: [3, 0] is not in list
```

## Why This Is A Bug

1. **Poor API design**: The method accepts invalid parameter values without validation, then fails with an unhelpful error message during internal encoding.

2. **Inconsistent with `set_job()`**: The similar method `set_job()` silently ignores invalid parameter values (e.g., `fit_type=5`), but `set_iprint()` raises an exception.

3. **Violates documentation**: The docstring states "The permissible values are 0, 1, and 2" but provides no indication that other values will cause a ValueError during encoding.

4. **Unclear error message**: The error message `"[3, 0] is not in list"` exposes internal implementation details and doesn't explain what the user did wrong.

## Fix

The method should validate parameters before attempting to encode them. Here's a patch:

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1009,6 +1009,17 @@ class ODR:
     def set_iprint(self, init=None, so_init=None,
         iter=None, so_iter=None, iter_step=None, final=None, so_final=None):
         """ Set the iprint parameter for the printing of computation reports.
+
+        # Validate inputs
+        valid_report_values = (None, 0, 1, 2)
+        if init not in valid_report_values:
+            raise ValueError(f"init must be 0, 1, or 2, got {init}")
+        if so_init not in valid_report_values:
+            raise ValueError(f"so_init must be 0, 1, or 2, got {so_init}")
+        if iter not in valid_report_values:
+            raise ValueError(f"iter must be 0, 1, or 2, got {iter}")
+        if so_iter not in valid_report_values:
+            raise ValueError(f"so_iter must be 0, 1, or 2, got {so_iter}")
+        if final not in valid_report_values:
+            raise ValueError(f"final must be 0, 1, or 2, got {final}")
+        if so_final not in valid_report_values:
+            raise ValueError(f"so_final must be 0, 1, or 2, got {so_final}")
+        if iter_step is not None and iter_step not in range(10):
+            raise ValueError(f"iter_step must be in range 0-9, got {iter_step}")

         If any of the arguments are specified here, then they are set in the
         iprint member. If iprint is not set manually or with this method, then
```

Alternatively, for consistency with `set_job()`, the method could silently ignore invalid values:

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1062,15 +1062,23 @@ class ODR:
         iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]

         if init is not None:
-            iprint_l[0] = init
+            if init in (0, 1, 2):
+                iprint_l[0] = init
         if so_init is not None:
-            iprint_l[1] = so_init
+            if so_init in (0, 1, 2):
+                iprint_l[1] = so_init
         if iter is not None:
-            iprint_l[2] = iter
+            if iter in (0, 1, 2):
+                iprint_l[2] = iter
         if so_iter is not None:
-            iprint_l[3] = so_iter
+            if so_iter in (0, 1, 2):
+                iprint_l[3] = so_iter
         if final is not None:
-            iprint_l[4] = final
+            if final in (0, 1, 2):
+                iprint_l[4] = final
         if so_final is not None:
-            iprint_l[5] = so_final
+            if so_final in (0, 1, 2):
+                iprint_l[5] = so_final
```