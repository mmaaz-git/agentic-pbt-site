# Bug Report: dask.dataframe.tseries.resample Division Length Mismatch

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function returns tuples `newdivs` and `outdivs` with different lengths when certain combinations of parameters are used. This causes failures in real resampling operations with the error "Index is not contained within new index."

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@settings(max_examples=1000)
@given(
    st.integers(min_value=2, max_value=20),
    st.sampled_from(['1h', '2h', '1D', '2D', '1W', '1M', '1Q', '1Y']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right'])
)
def test_resample_bin_and_out_divs_equal_length(n_divisions, rule, closed, label):
    start = pd.Timestamp('2020-01-01')
    divisions = pd.date_range(start, periods=n_divisions, freq='1D')

    newdivs, outdivs = _resample_bin_and_out_divs(
        divisions, rule, closed=closed, label=label
    )

    assert len(newdivs) == len(outdivs), \
        f"newdivs and outdivs must have same length: {len(newdivs)} != {len(outdivs)}"
```

**Failing input**: `n_divisions=3, rule='2D', closed='right', label='right'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

start = pd.Timestamp('2020-01-01')
divisions = pd.date_range(start, periods=3, freq='1D')

newdivs, outdivs = _resample_bin_and_out_divs(
    divisions, rule='2D', closed='right', label='right'
)

print(f"newdivs length: {len(newdivs)}")
print(f"outdivs length: {len(outdivs)}")
print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")
```

Output:
```
newdivs length: 3
outdivs length: 2
newdivs: (Timestamp('2020-01-01 00:00:00'), Timestamp('2020-01-01 00:00:00.000000001'), Timestamp('2020-01-03 00:00:00.000000001'))
outdivs: (Timestamp('2020-01-01 00:00:00'), Timestamp('2020-01-03 00:00:00'))
```

This also causes failures in actual resampling operations:

```python
import pandas as pd
import dask.dataframe as dd

dates = pd.date_range('2020-01-01', periods=100, freq='1D')
df = pd.DataFrame({'value': range(100)}, index=dates)
ddf = dd.from_pandas(df, npartitions=3)

result = ddf.resample('2D', closed='right', label='right').sum()
computed = result.compute()
```

This raises: `ValueError: Index is not contained within new index.`

## Why This Is A Bug

The function `_resample_bin_and_out_divs` is supposed to return paired division boundaries. The `newdivs` represent the bin divisions used for partitioning, while `outdivs` represent the output divisions. These must have the same length because they represent start/end pairs for each partition.

The root cause is in lines 93-104 of `resample.py`. The code uses a single `setter` lambda function for both `newdivs` and `outdivs`, but the choice between `append` or `__setitem__` is based only on the length of `newdivs`:

```python
if len(newdivs) < len(divs):
    setter = lambda a, val: a.append(val)
else:
    setter = lambda a, val: a.__setitem__(-1, val)
```

This means:
- When `len(newdivs) < len(divs)`, the setter appends
- This setter is then used on both `newdivs` and `outdivs`
- If only `newdivs` needs adjustment but not `outdivs`, they end up with different lengths

## Fix

```diff
--- a/resample.py
+++ b/resample.py
@@ -90,16 +90,19 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
     # Adjust ends
     if newdivs[0] < divisions[0]:
         newdivs[0] = divisions[0]
     if newdivs[-1] < divisions[-1]:
-        if len(newdivs) < len(divs):
-            setter = lambda a, val: a.append(val)
+        if len(newdivs) < len(divs):
+            newdivs.append(divisions[-1] + res)
         else:
-            setter = lambda a, val: a.__setitem__(-1, val)
-        setter(newdivs, divisions[-1] + res)
+            newdivs[-1] = divisions[-1] + res
+
         if outdivs[-1] > divisions[-1]:
-            setter(outdivs, outdivs[-1])
+            if len(outdivs) < len(divs):
+                outdivs.append(outdivs[-1])
+            else:
+                outdivs[-1] = outdivs[-1]
         elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            if len(outdivs) < len(divs):
+                outdivs.append(temp.index[-1])
+            else:
+                outdivs[-1] = temp.index[-1]

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```