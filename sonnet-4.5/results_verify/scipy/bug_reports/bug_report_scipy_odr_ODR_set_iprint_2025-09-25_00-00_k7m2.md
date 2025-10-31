# Bug Report: scipy.odr.ODR.set_iprint Crashes with IndexError/ValueError

**Target**: `scipy.odr.ODR.set_iprint`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ODR.set_iprint()` method crashes with `IndexError` or `ValueError` when the `iprint` attribute contains certain values. This occurs because the method attempts to decode `iprint` digits using a lookup table that only has 7 entries (indices 0-6), but `iprint` digits can be 0-9.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.odr import ODR, Data, Model
import pytest

def linear_func(B, x):
    return B[0] * x + B[1]

@given(st.integers(min_value=0, max_value=9999))
@settings(max_examples=200)
def test_set_iprint_robust_to_manual_iprint(manual_iprint):
    data = Data([1, 2, 3], [2, 4, 6])
    model = Model(linear_func)
    odr = ODR(data, model, beta0=[1, 1])

    odr.iprint = manual_iprint

    odr.set_iprint(final=0)
```

**Failing inputs**:
- `manual_iprint=67` (causes IndexError)
- `manual_iprint=3` (causes ValueError)

## Reproducing the Bug

```python
from scipy.odr import ODR, Data, Model

def linear_func(B, x):
    return B[0] * x + B[1]

data = Data([1, 2, 3], [2, 4, 6])
model = Model(linear_func)
odr = ODR(data, model, beta0=[1, 1])

odr.iprint = 67
odr.set_iprint(final=0)
```

This crashes with:
```
IndexError: list index out of range
  at line 1060: iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]
```

Alternative minimal example:
```python
odr.iprint = 3
odr.set_iprint(final=0)
```

This crashes with:
```
ValueError: [0, 1] is not in list
  at line 1081: ip[3] = ip2arg.index(iprint_l[4:6])
```

## Why This Is A Bug

The `set_iprint` method (lines 1009-1084 in `_odrpack.py`) has a fundamental flaw:

1. **Line 1038-1041**: It extracts digits from `iprint` (each digit can be 0-9):
   ```python
   ip = [self.iprint // 1000 % 10,  # Can be 0-9
         self.iprint // 100 % 10,   # Can be 0-9
         self.iprint // 10 % 10,    # Can be 0-9
         self.iprint % 10]          # Can be 0-9
   ```

2. **Line 1045-1051**: Defines `ip2arg` with only 7 entries (indices 0-6):
   ```python
   ip2arg = [[0, 0], [1, 0], [2, 0], [1, 1], [2, 1], [1, 2], [2, 2]]
   ```

3. **Line 1060**: Tries to index `ip2arg` with values from `ip`:
   ```python
   iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]
   ```

   **If `ip[0]`, `ip[1]`, or `ip[3]` is >= 7, this raises `IndexError`!**

4. **Lines 1079-1081**: Try to find values in `ip2arg`:
   ```python
   ip[0] = ip2arg.index(iprint_l[0:2])
   ip[1] = ip2arg.index(iprint_l[2:4])
   ip[3] = ip2arg.index(iprint_l[4:6])
   ```

   **If `iprint_l[0:2]`, `[2:4]`, or `[4:6]` is not in `ip2arg`, this raises `ValueError`!**

This can occur in normal usage if `iprint` was set manually or through other code paths that don't validate the encoded value.

## Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -1057,7 +1057,17 @@ class ODR:
             raise OdrError(
                 "no rptfile specified, cannot output to stdout twice")

-        iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]
+        # Validate that iprint digits are within valid range
+        for i, digit in enumerate([ip[0], ip[1], ip[3]]):
+            if digit < 0 or digit >= len(ip2arg):
+                raise ValueError(
+                    f"Invalid iprint encoding: digit at position {i} is {digit}, "
+                    f"but must be in range [0, {len(ip2arg)-1}]"
+                )
+
+        try:
+            iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]
+        except (IndexError, KeyError) as e:
+            raise ValueError(f"Invalid iprint value {self.iprint}: {e}")

         if init is not None:
+            if init not in (0, 1, 2):
+                raise ValueError(f"init must be 0, 1, or 2, got {init}")
             iprint_l[0] = init
         if so_init is not None:
+            if so_init not in (0, 1, 2):
+                raise ValueError(f"so_init must be 0, 1, or 2, got {so_init}")
             iprint_l[1] = so_init
         if iter is not None:
+            if iter not in (0, 1, 2):
+                raise ValueError(f"iter must be 0, 1, or 2, got {iter}")
             iprint_l[2] = iter
         if so_iter is not None:
+            if so_iter not in (0, 1, 2):
+                raise ValueError(f"so_iter must be 0, 1, or 2, got {so_iter}")
             iprint_l[3] = so_iter
         if final is not None:
+            if final not in (0, 1, 2):
+                raise ValueError(f"final must be 0, 1, or 2, got {final}")
             iprint_l[4] = final
         if so_final is not None:
+            if so_final not in (0, 1, 2):
+                raise ValueError(f"so_final must be 0, 1, or 2, got {so_final}")
             iprint_l[5] = so_final

         if iter_step in range(10):
             ip[2] = iter_step

-        ip[0] = ip2arg.index(iprint_l[0:2])
-        ip[1] = ip2arg.index(iprint_l[2:4])
-        ip[3] = ip2arg.index(iprint_l[4:6])
+        try:
+            ip[0] = ip2arg.index(iprint_l[0:2])
+            ip[1] = ip2arg.index(iprint_l[2:4])
+            ip[3] = ip2arg.index(iprint_l[4:6])
+        except ValueError as e:
+            raise ValueError(
+                f"Invalid combination of iprint parameters: {e}. "
+                f"Valid combinations are: {ip2arg}"
+            )

         self.iprint = ip[0]*1000 + ip[1]*100 + ip[2]*10 + ip[3]
```