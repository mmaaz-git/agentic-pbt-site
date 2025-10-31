# Bug Report: pandas.core.ops.kleene_xor Commutativity Violation

**Target**: `pandas.core.ops.kleene_xor`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `kleene_xor` function violates commutativity by returning different result values for `X ^ NA` vs `NA ^ X` when both are masked (NA), contradicting the explicit claim in the code comment that "A ^ B == B ^ A".

## Property-Based Test

```python
import numpy as np
from pandas._libs import missing as libmissing
import pandas.core.ops as ops
from hypothesis import given, strategies as st, settings


@given(left_val=st.booleans())
@settings(max_examples=10)
def test_kleene_xor_na_commutativity_full(left_val):
    left_with_value = np.array([left_val])
    mask_with_value = np.array([False])

    left_with_na = np.array([False])
    mask_with_na = np.array([True])

    result_value_na, mask_value_na = ops.kleene_xor(left_with_value, libmissing.NA, mask_with_value, None)
    result_na_value, mask_na_value = ops.kleene_xor(left_with_na, left_val, mask_with_na, None)

    assert mask_value_na[0] == mask_na_value[0]
    assert result_value_na[0] == result_na_value[0]
```

**Failing input**: `left_val=True`

## Reproducing the Bug

```python
import numpy as np
from pandas._libs import missing as libmissing
import pandas.core.ops as ops

left_true = np.array([True])
mask_false = np.array([False])

left_na = np.array([False])
mask_true = np.array([True])

result1, mask1 = ops.kleene_xor(left_true, libmissing.NA, mask_false, None)
result2, mask2 = ops.kleene_xor(left_na, True, mask_true, None)

print(f'True ^ NA: result={result1[0]}, mask={mask1[0]}')
print(f'NA ^ True: result={result2[0]}, mask={mask2[0]}')

assert result1[0] == result2[0]
```

## Why This Is A Bug

The function contains an explicit comment claiming commutativity: "A ^ B == B ^ A", and the code even swaps arguments to ensure this property. However, when one operand is NA, the result values differ depending on argument order:

- `True ^ NA` returns `(result=False, mask=True)`
- `NA ^ True` returns `(result=True, mask=True)`

While both correctly indicate NA via `mask=True`, the underlying result values are inconsistent. This violates the documented commutativity property and could cause subtle bugs if any code inspects result values even when masked.

## Fix

```diff
--- a/pandas/core/ops/__init__.py
+++ b/pandas/core/ops/__init__.py
@@ -172,7 +172,7 @@ def kleene_xor(
     raise_for_nan(right, method="xor")
     if right is libmissing.NA:
-        result = np.zeros_like(left)
+        result = left.copy()
     else:
         result = left ^ right
```

This ensures that when `right=NA`, the result computation uses the left values (just like when `right` is a boolean), making the operation fully commutative.