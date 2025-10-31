# Bug Report: pandas.core.ops._masked_arith_op rpow with base=1

**Target**: `pandas.core.ops.array_ops._masked_arith_op`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When using `_masked_arith_op` with the `rpow` operator and a base of 1, the function incorrectly returns NaN instead of 1.0. Since 1**x = 1 for any value of x (including NaN), the result should always be 1.0, not NaN.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core import roperator
from pandas.core.ops.array_ops import _masked_arith_op


@settings(max_examples=500)
@given(
    exponent=st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10)
)
def test_masked_arith_op_rpow_base_one_should_return_one(exponent):
    x = np.array([exponent, np.nan, exponent], dtype=object)
    y = 1.0

    result = _masked_arith_op(x, y, roperator.rpow)

    assert result[0] == 1.0
    assert result[2] == 1.0
```

**Failing input**: `exponent=0.0`

## Reproducing the Bug

```python
import numpy as np
from pandas.core import roperator
from pandas.core.ops.array_ops import _masked_arith_op

x = np.array([0.0, 1.0, 2.0], dtype=object)
y = 1.0

result = _masked_arith_op(x, y, roperator.rpow)

print(f"Result: {result}")
print(f"Expected: [1.0, 1.0, 1.0]")
```

Expected: `[1.0, 1.0, 1.0]` (since 1.0**0.0 = 1.0, 1.0**1.0 = 1.0, 1.0**2.0 = 1.0)

Actual: `[nan, nan, nan]`

## Why This Is A Bug

In mathematics and NumPy, `1 ** x = 1` for any value of x, including NaN:
```python
>>> 1.0 ** np.nan
1.0
>>> 1.0 ** 1000.0
1.0
```

The function `_masked_arith_op` has special handling for this case with a comment on line 175: "1 ** np.nan is 1. So we have to unmask those."

However, the implementation on line 179 is incorrect:
```python
elif op is roperator.rpow:
    mask = np.where(y == 1, False, mask)
```

This sets `mask` to `False` where `y == 1`, which prevents the operation from being computed. Instead, NaN is filled in these positions (line 184). The mask should remain `True` (or the result should be directly set to 1.0) to preserve the mathematical property that 1**x = 1.

## Fix

```diff
--- a/pandas/core/ops/array_ops.py
+++ b/pandas/core/ops/array_ops.py
@@ -175,8 +175,12 @@ def _masked_arith_op(x: np.ndarray, y, op):
         # 1 ** np.nan is 1. So we have to unmask those.
         if op is pow:
             mask = np.where(x == 1, False, mask)
+            result[x == 1] = 1
         elif op is roperator.rpow:
-            mask = np.where(y == 1, False, mask)
+            if y == 1:
+                result[:] = 1
+                mask = np.zeros_like(mask, dtype=bool)
+                return result.reshape(x.shape)

         if mask.any():
             result[mask] = op(xrav[mask], y)
```

Alternative simpler fix:

```diff
--- a/pandas/core/ops/array_ops.py
+++ b/pandas/core/ops/array_ops.py
@@ -175,8 +175,8 @@ def _masked_arith_op(x: np.ndarray, y, op):
         # 1 ** np.nan is 1. So we have to unmask those.
         if op is pow:
-            mask = np.where(x == 1, False, mask)
+            pass  # Keep mask as is, numpy handles 1**nan correctly
         elif op is roperator.rpow:
-            mask = np.where(y == 1, False, mask)
+            pass  # Keep mask as is, numpy handles 1**x correctly
```

Actually, the simplest fix is to just remove these lines entirely, since NumPy already handles 1**nan = 1 correctly when we compute the operation normally.