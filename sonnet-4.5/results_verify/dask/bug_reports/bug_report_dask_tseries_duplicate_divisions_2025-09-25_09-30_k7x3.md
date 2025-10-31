# Bug Report: dask.dataframe.tseries Duplicate Divisions

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function can produce output divisions (outdivs) with duplicate consecutive elements due to a logic error on line 99 where `setter(outdivs, outdivs[-1])` either creates a duplicate by appending the last element again, or performs a no-op by setting the last element to itself.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs
from hypothesis import given, strategies as st, settings, assume

@given(
    st.lists(st.datetimes(min_value=pd.Timestamp('2000-01-01'),
                          max_value=pd.Timestamp('2025-12-31')),
             min_size=2, max_size=20, unique=True).map(sorted),
    st.sampled_from(['D', 'h', '30min', 'W', 'ME']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right'])
)
@settings(max_examples=500)
def test_no_duplicate_divisions(divisions_list, rule, closed, label):
    divisions = pd.DatetimeIndex(divisions_list)

    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)
    except Exception:
        assume(False)

    for divs_name, divs in [('newdivs', newdivs), ('outdivs', outdivs)]:
        for i in range(len(divs) - 1):
            assert divs[i] != divs[i+1], (
                f"{divs_name} has duplicate consecutive elements at index {i}: "
                f"{divs[i]} == {divs[i+1]}"
            )
```

**Failing input**:
```python
divisions_list=[datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 1, 0, 0, 0, 1)]
rule='D'
closed='left'
label='left'
```

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import pandas as pd
from datetime import datetime
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

divisions = pd.DatetimeIndex([
    datetime(2000, 1, 1, 0, 0, 0, 0),
    datetime(2000, 1, 1, 0, 0, 0, 1)
])

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'D', closed='left', label='left')

print("outdivs:", outdivs)

for i in range(len(outdivs) - 1):
    if outdivs[i] == outdivs[i+1]:
        print(f"BUG: outdivs[{i}] == outdivs[{i+1}] == {outdivs[i]}")
```

## Why This Is A Bug

Divisions in Dask DataFrames must be strictly monotonically increasing without duplicates. They represent partition boundaries, and having duplicate consecutive divisions violates this invariant. This can cause incorrect behavior in operations that rely on divisions being unique.

The bug occurs in the division adjustment logic at lines 92-101 in `resample.py`:

```python
if newdivs[-1] < divisions[-1]:
    if len(newdivs) < len(divs):
        setter = lambda a, val: a.append(val)
    else:
        setter = lambda a, val: a.__setitem__(-1, val)
    setter(newdivs, divisions[-1] + res)
    if outdivs[-1] > divisions[-1]:
        setter(outdivs, outdivs[-1])  # BUG: Line 99
    elif outdivs[-1] < divisions[-1]:
        setter(outdivs, temp.index[-1])
```

The `setter` lambda is chosen based on `len(newdivs) < len(divs)`, but is then used for both `newdivs` and `outdivs`. When the condition is true:
- `setter` uses `append`
- Line 99: `setter(outdivs, outdivs[-1])` becomes `outdivs.append(outdivs[-1])`, creating a duplicate

When the condition is false:
- `setter` uses `__setitem__(-1, val)`
- Line 99: `setter(outdivs, outdivs[-1])` becomes `outdivs[-1] = outdivs[-1]`, a no-op

## Fix

The issue is that `newdivs` and `outdivs` may have different lengths, so they need separate logic for whether to append or set the last element. Here's the fix:

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -96,7 +96,11 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
             setter = lambda a, val: a.__setitem__(-1, val)
         setter(newdivs, divisions[-1] + res)
         if outdivs[-1] > divisions[-1]:
-            setter(outdivs, outdivs[-1])
+            if len(outdivs) < len(divs):
+                pass
+            else:
+                outdivs[-1] = divisions[-1]
         elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            if len(outdivs) < len(divs):
+                outdivs.append(temp.index[-1])
+            else:
+                outdivs[-1] = temp.index[-1]
```