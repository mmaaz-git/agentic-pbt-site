# Bug Report: scipy.odr.ODR.set_iprint Missing Input Validation

**Target**: `scipy.odr.ODR.set_iprint`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ODR.set_iprint` method lacks input validation for its parameters (`init`, `so_init`, `iter`, `so_iter`, `final`, `so_final`), causing a confusing `ValueError` when invalid values are passed. According to the docstring, these parameters should be 0, 1, or 2, but the method doesn't enforce this constraint.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.odr import Data, ODR, unilinear
import tempfile

def make_odr():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    data = Data(x, y)
    return ODR(data, unilinear, beta0=[1.0, 0.0])

@given(
    init=st.integers(min_value=-5, max_value=10),
    iter_param=st.integers(min_value=-5, max_value=10),
    final=st.integers(min_value=-5, max_value=10)
)
@settings(max_examples=200)
def test_set_iprint_validates_inputs(init, iter_param, final):
    odr_obj = make_odr()
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        odr_obj.rptfile = f.name

    try:
        odr_obj.set_iprint(init=init, iter=iter_param, final=final)
    except ValueError as e:
        if "is not in list" in str(e):
            raise AssertionError(f"Missing input validation: {e}")
```

**Failing input**: `init=3` (or any value outside [0, 1, 2])

## Reproducing the Bug

```python
import numpy as np
from scipy.odr import Data, ODR, unilinear
import tempfile

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
odr_obj = ODR(data, unilinear, beta0=[1.0, 0.0])

with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    odr_obj.rptfile = f.name

odr_obj.set_iprint(init=3)
```

**Expected behavior**: Raises a clear `ValueError` like "init must be 0, 1, or 2, got 3"

**Actual behavior**: Raises `ValueError: [3, 0] is not in list` from the internal `.index()` call at line 1079

## Why This Is A Bug

The docstring for `set_iprint` (lines 1022-1025 in `_odrpack.py`) states:

```
There are three reports: initialization, iteration, and final reports.
They are represented by the arguments init, iter, and final
respectively. The permissible values are 0, 1, and 2 representing "no
report", "short report", and "long report" respectively.
```

The method should validate these constraints but doesn't. Instead, invalid values cause an internal implementation error when trying to find the constructed list in the `ip2arg` lookup table (line 1079):

```python
ip[0] = ip2arg.index(iprint_l[0:2])  # Fails if iprint_l[0:2] not in ip2arg
```

This violates the principle of fail-fast with clear error messages. Users get a confusing error about list membership rather than being told their parameter value is invalid.

## Fix

Add input validation at the beginning of `set_iprint`:

```diff
diff --git a/scipy/odr/_odrpack.py b/scipy/odr/_odrpack.py
index 1234567..abcdefg 100644
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1034,6 +1034,24 @@ class ODR:
         exception.
         """
+        # Validate input parameters
+        for param_name, param_value in [('init', init), ('so_init', so_init),
+                                         ('iter', iter), ('so_iter', so_iter),
+                                         ('final', final), ('so_final', so_final)]:
+            if param_value is not None and param_value not in (0, 1, 2):
+                raise ValueError(
+                    f"{param_name} must be 0, 1, or 2, got {param_value}"
+                )
+
+        # Validate iter_step
+        if iter_step is not None and iter_step not in range(10):
+            raise ValueError(
+                f"iter_step must be between 0 and 9, got {iter_step}"
+            )
+
+        # Validate that stdout printing only works with rptfile
+        if (self.rptfile is None and
+            (so_init is not None or so_iter is not None or so_final is not None)):
+            raise OdrError(
+                "no rptfile specified, cannot output to stdout twice")
+
         if self.iprint is None:
             self.iprint = 0

@@ -1051,12 +1069,6 @@ class ODR:
                   [1, 2],  # short, long
                   [2, 2]]  # long,  long

-        if (self.rptfile is None and
-            (so_init is not None or
-             so_iter is not None or
-             so_final is not None)):
-            raise OdrError(
-                "no rptfile specified, cannot output to stdout twice")
-
         iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]

         if init is not None:
```

The key changes:
1. Validate all parameters before using them
2. Move the rptfile validation to the top (before any modifications)
3. Provide clear, user-friendly error messages