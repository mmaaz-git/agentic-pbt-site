# Bug Report: dask.dataframe.dask_expr Rolling Window with Timedelta and center=True

**Target**: `dask.dataframe.dask_expr._rolling.RollingReduction._lower`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`DataFrame.rolling()` crashes with `TypeError` when using a string/timedelta window (e.g., '2h') with `center=True`. The code attempts integer division on a string, which pandas supports but dask does not handle correctly.

## Property-Based Test

```python
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings
from dask.dataframe import from_pandas


@given(
    window_value=st.sampled_from(['1h', '2h', '1D', '30min']),
    center=st.booleans(),
    npartitions=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=50)
def test_rolling_window_type_compatibility(window_value, center, npartitions):
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=20, freq='30min'),
        'value': range(20)
    })
    df = df.set_index('time')

    ddf = from_pandas(df, npartitions=npartitions)

    pandas_result = df.rolling(window=window_value, center=center).mean()

    dask_result = ddf.rolling(window=window_value, center=center).mean()
    dask_computed = dask_result.compute()

    pd.testing.assert_frame_equal(
        dask_computed.sort_index(),
        pandas_result.sort_index(),
        check_dtype=False
    )
```

**Failing input**: `window_value='1h', center=True, npartitions=2`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe import from_pandas

df = pd.DataFrame({
    'time': pd.date_range('2020-01-01', periods=10, freq='1h'),
    'value': range(10)
})
df = df.set_index('time')

ddf = from_pandas(df, npartitions=2)

result = ddf.rolling(window='2h', center=True).mean()
computed = result.compute()
```

**Error:**
```
TypeError: unsupported operand type(s) for //: 'str' and 'int'
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_rolling.py", line 128, in _lower
    before = self.window // 2
             ^^^^^^^^^^^^^^^^
```

## Why This Is A Bug

Pandas supports `rolling(window='2h', center=True)` on datetime-indexed DataFrames, but Dask crashes when attempting the same operation. The code in `_rolling.py:128` attempts integer division (`//`) on the window parameter without checking if it's an integer first.

## Fix

The `_lower` method in `RollingReduction` needs to check the window type before performing integer operations:

```diff
--- a/_rolling.py
+++ b/_rolling.py
@@ -126,7 +126,7 @@ class RollingReduction(Expr):

         if self.kwargs.get("center"):
-            before = self.window // 2
-            after = self.window - before - 1
+            if isinstance(self.window, int):
+                before = self.window // 2
+                after = self.window - before - 1
+            else:
+                before = pd.Timedelta(self.window) / 2
+                after = before
         elif not isinstance(self.window, int):
             before = pd.Timedelta(self.window)
             after = 0
```