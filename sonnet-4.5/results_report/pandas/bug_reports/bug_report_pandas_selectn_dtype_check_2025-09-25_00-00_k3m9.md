# Bug Report: pandas.core.methods.selectn dtype check order

**Target**: `pandas.core.methods.selectn.SelectNSeries.compute`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`nlargest(0)` and `nsmallest(0)` raise `TypeError` on Series with object dtype, even though these operations don't require comparison and should return empty Series regardless of dtype.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st


@given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20))
def test_nlargest_zero_works_on_any_dtype(data):
    s = pd.Series(data)
    result = s.nlargest(0)
    assert len(result) == 0
```

**Failing input**: `data=['0']`

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['a', 'b', 'c'])

result = s.nlargest(0)
```

**Output**:
```
TypeError: Cannot use method 'nlargest' with dtype object
```

## Why This Is A Bug

The code in `selectn.py:90-99` checks dtype validity (line 95-96) BEFORE checking if `n <= 0` (line 98-99):

```python
def compute(self, method: str) -> Series:
    n = self.n
    dtype = self.obj.dtype
    if not self.is_valid_dtype_n_method(dtype):  # Line 95-96
        raise TypeError(f"Cannot use method '{method}' with dtype {dtype}")

    if n <= 0:  # Line 98-99
        return self.obj[[]]
```

When `n <= 0`, the method returns an empty Series without performing any comparisons. Therefore, dtype validation is unnecessary in this case. The dtype check prevents reaching the `n <= 0` early return, causing valid operations to fail.

## Fix

```diff
--- a/pandas/core/methods/selectn.py
+++ b/pandas/core/methods/selectn.py
@@ -92,12 +92,13 @@ class SelectNSeries(SelectN):

         n = self.n
         dtype = self.obj.dtype
+
+        if n <= 0:
+            return self.obj[[]]
+
         if not self.is_valid_dtype_n_method(dtype):
             raise TypeError(f"Cannot use method '{method}' with dtype {dtype}")

-        if n <= 0:
-            return self.obj[[]]
-
         dropped = self.obj.dropna()
         nan_index = self.obj.drop(dropped.index)
```