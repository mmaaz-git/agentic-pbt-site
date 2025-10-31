# Bug Report: pd.qcut Crash with Denormal Floats

**Target**: `pandas.qcut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.qcut` crashes with a confusing ValueError about "missing values" when binning data containing denormal floating-point numbers, even though the input contains no missing values.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
import pandas as pd


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=50),
    q=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_qcut_assigns_all_values(values, q):
    assume(len(set(values)) >= q)
    try:
        result = pd.qcut(values, q=q)
        assert len(result) == len(values)
        non_null = result.notna().sum()
        assert non_null == len(values)
    except ValueError as e:
        if "Bin edges must be unique" in str(e):
            assume(False)
        raise
```

**Failing input**: `values=[0.0, 2.458649946504791e-307], q=2`

## Reproducing the Bug

```python
import pandas as pd

values = [0.0, 2.458649946504791e-307]
result = pd.qcut(values, q=2)
```

Output:
```
ValueError: missing values must be missing in the same location both left and right sides
```

Full traceback shows the error originates from `pandas/core/arrays/interval.py:664` in the `_validate` method of IntervalArray, called during the qcut binning process.

## Why This Is A Bug

The `pd.qcut` function should accept any valid numeric array. The docstring specifies:

> "x : 1d ndarray or Series"

The input `[0.0, 2.458649946504791e-307]` is a valid 1D array of floats. However, the function crashes with a misleading error message about "missing values" when the actual problem is numerical precision issues during quantile computation. The value 2.458649946504791e-307 is a denormal (subnormal) float, which is a valid IEEE 754 number.

This affects users working with very small measurements (e.g., physical constants, molecular concentrations) and makes the error message unhelpful for debugging.

## Fix

Similar to `pd.cut`, the issue stems from numerical precision problems when calculating bin edges for denormal floats. The fix should:

1. Detect when quantile calculation produces invalid bin edges
2. Provide a clear error message about the value range issue
3. Or use higher-precision arithmetic to handle denormal floats

A defensive fix:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -310,6 +310,13 @@ def qcut(
     if is_integer(q):
         quantiles = np.linspace(0, 1, q + 1)
     else:
         quantiles = q
+
+    bins_result = np.nanquantile(x_idx, quantiles)
+    if len(bins_result) >= 2:
+        value_range = bins_result[-1] - bins_result[0]
+        if value_range > 0 and value_range < 1e-300:
+            raise ValueError(
+                f"Cannot create quantile bins for extremely small value range. "
+                "Consider scaling your data.")
     bins = np.nanquantile(x_idx, quantiles)
```