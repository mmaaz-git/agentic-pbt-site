# Bug Report: scipy.odr.ODR.set_iprint Crashes with ValueError

**Target**: `scipy.odr.ODR.set_iprint`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `set_iprint` method crashes with `ValueError: [0, 1] is not in list` when attempting to set certain valid combinations of print settings, specifically when trying to print to stdout without printing to file.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy import odr
import numpy as np

@settings(max_examples=100)
@given(
    init_val=st.integers(min_value=0, max_value=2),
    so_init_val=st.integers(min_value=0, max_value=2)
)
def test_set_iprint_no_crash(init_val, so_init_val):
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    data = odr.Data(x, y)
    model = odr.unilinear
    odr_obj = odr.ODR(data, model, beta0=[1.0, 1.0], rptfile='test.txt')

    odr_obj.set_iprint(init=init_val, so_init=so_init_val)
```

**Failing input**: `init=0, so_init=1`

## Reproducing the Bug

```python
import numpy as np
from scipy import odr

x = np.array([0.0, 1.0, 2.0])
y = np.array([0.0, 1.0, 2.0])
data = odr.Data(x, y)
model = odr.unilinear
odr_obj = odr.ODR(data, model, beta0=[1.0, 1.0], rptfile='test_report.txt')

odr_obj.set_iprint(init=0, so_init=1)
```

## Why This Is A Bug

The `ip2arg` lookup list is missing the combinations `[0, 1]` and `[0, 2]`, which would represent printing to stdout without printing to file. While the docstring mentions "one cannot specify to print to stdout but not a file," the code should handle this case gracefully with a proper error message (as it attempts to do at lines 1053-1058), not crash with a cryptic ValueError during the index lookup.

The check at lines 1053-1058 tries to catch this case, but it only checks if `rptfile is None`. When `rptfile` is set but the user tries to set `init=0, so_init=1`, the code bypasses this check and crashes later at line 1079 when it tries to find `[0, 1]` in `ip2arg`.

## Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1043,7 +1043,9 @@ class ODR:
         # make a list to convert iprint digits to/from argument inputs
         #                   rptfile, stdout
         ip2arg = [[0, 0],  # none,  none
+                  [0, 1],  # none,  short
+                  [0, 2],  # none,  long
                   [1, 0],  # short, none
                   [2, 0],  # long,  none
                   [1, 1],  # short, short
@@ -1051,11 +1053,23 @@ class ODR:
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
@@ -1075,6 +1089,14 @@ class ODR:
         if iter_step in range(10):
             # 0..9
             ip[2] = iter_step
+
+        # Check for invalid combinations
+        for i, pair in enumerate([iprint_l[0:2], iprint_l[2:4], iprint_l[4:6]]):
+            if pair not in ip2arg:
+                raise OdrError(
+                    f"Invalid print setting combination: cannot print to stdout "
+                    f"without also printing to file. Got {pair}."
+                )

         ip[0] = ip2arg.index(iprint_l[0:2])
         ip[1] = ip2arg.index(iprint_l[2:4])