# Bug Report: dask.dataframe.tseries Division Length Mismatch

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function can produce `newdivs` and `outdivs` with different lengths, caused by inconsistent application of the append operation in the boundary adjustment logic.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@given(
    st.integers(min_value=2, max_value=20),
    st.sampled_from(['h', 'D', '2h', '3D', '12h', 'W']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right']),
)
@settings(max_examples=500)
def test_resample_divisions_same_length(n_divs, freq, closed, label):
    start = pd.Timestamp('2000-01-01')
    end = start + pd.Timedelta(days=30)
    divisions = pd.date_range(start, end, periods=n_divs)

    newdivs, outdivs = _resample_bin_and_out_divs(divisions, freq, closed=closed, label=label)

    assert len(newdivs) == len(outdivs), f"newdivs length {len(newdivs)} != outdivs length {len(outdivs)}"
```

**Failing input**: `n_divs=12, freq='3D', closed='right', label='right'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

start = pd.Timestamp('2000-01-01')
end = start + pd.Timedelta(days=30)
divisions = pd.date_range(start, end, periods=12)

newdivs, outdivs = _resample_bin_and_out_divs(divisions, '3D', closed='right', label='right')

print(f"newdivs length: {len(newdivs)}")
print(f"outdivs length: {len(outdivs)}")

if len(newdivs) != len(outdivs):
    print(f"ERROR: Length mismatch!")
    print(f"  newdivs ({len(newdivs)}): {newdivs}")
    print(f"  outdivs ({len(outdivs)}): {outdivs}")
```

## Why This Is A Bug

The boundary adjustment logic (lines 92-101) uses a single `setter` function for both `newdivs` and `outdivs`, chosen based on whether `len(newdivs) < len(divs)`. However, the logic applies the setter differently:

1. Line 97: Always applies `setter(newdivs, divisions[-1] + res)` when `newdivs[-1] < divisions[-1]`
2. Lines 98-101: Conditionally applies setter to `outdivs` based on whether `outdivs[-1] > divisions[-1]` or `outdivs[-1] < divisions[-1]`

When `setter` is the `append` lambda (triggered when `len(newdivs) < len(divs)`), this causes:
- `newdivs` to unconditionally get one element appended
- `outdivs` to conditionally get zero or one element appended (or neither if both conditions are false)

This results in `newdivs` and `outdivs` having different lengths, which violates the expected invariant that these division tuples should be parallel structures representing the same partitioning scheme with different labeling.

## Fix

The fix requires ensuring both `newdivs` and `outdivs` are adjusted consistently. One approach is to determine if an append is needed upfront, and apply it to both:

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -91,14 +91,18 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         newdivs[0] = divisions[0]
     if newdivs[-1] < divisions[-1]:
         if len(newdivs) < len(divs):
-            setter = lambda a, val: a.append(val)
+            newdivs.append(divisions[-1] + res)
+            if outdivs[-1] < divisions[-1]:
+                outdivs.append(temp.index[-1])
+            else:
+                outdivs.append(outdivs[-1])
         else:
-            setter = lambda a, val: a.__setitem__(-1, val)
-        setter(newdivs, divisions[-1] + res)
-        if outdivs[-1] > divisions[-1]:
-            setter(outdivs, outdivs[-1])
-        elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            newdivs[-1] = divisions[-1] + res
+            if outdivs[-1] < divisions[-1]:
+                outdivs[-1] = temp.index[-1]
+            else:
+                pass

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```

This fix separates the append and setitem paths, ensuring both `newdivs` and `outdivs` are adjusted in the same manner (both append or both setitem), maintaining equal lengths.