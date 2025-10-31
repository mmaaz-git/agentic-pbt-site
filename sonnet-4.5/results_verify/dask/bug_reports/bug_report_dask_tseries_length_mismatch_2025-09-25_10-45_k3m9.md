# Bug Report: dask.dataframe.tseries Length Mismatch

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function returns `newdivs` and `outdivs` tuples with mismatched lengths under certain conditions, violating the invariant that these two outputs should always have the same length.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@given(
    st.integers(min_value=2, max_value=100),
    st.sampled_from(['h', 'D', '2h', '30min', 'W', 'M', 'Q', 'Y']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right'])
)
@settings(max_examples=1000)
def test_resample_newdivs_outdivs_length_consistency(n_divisions, rule, closed, label):
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2023-12-31')
    divisions = pd.date_range(start, end, periods=n_divisions)

    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)
    except Exception as e:
        assume(False)

    assert len(newdivs) == len(outdivs), \
        f"Length mismatch: newdivs={len(newdivs)}, outdivs={len(outdivs)}"
```

**Failing input**: `n_divisions=5, rule='Y', closed='right', label='right'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

start = pd.Timestamp('2020-01-01')
end = pd.Timestamp('2023-12-31')
divisions = pd.date_range(start, end, periods=5)

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'Y', closed='right', label='right')

print(f"newdivs length: {len(newdivs)}")
print(f"outdivs length: {len(outdivs)}")
print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")

assert len(newdivs) == len(outdivs)
```

Output:
```
newdivs length: 5
outdivs length: 4
newdivs: (Timestamp('2020-01-01 00:00:00'), Timestamp('2021-01-01 00:00:00'), Timestamp('2022-01-01 00:00:00'), Timestamp('2023-01-01 00:00:00'), Timestamp('2024-01-01 00:00:00'))
outdivs: (Timestamp('2020-12-31 00:00:00'), Timestamp('2021-12-31 00:00:00'), Timestamp('2022-12-31 00:00:00'), Timestamp('2023-12-31 00:00:00'))
AssertionError
```

## Why This Is A Bug

The function `_resample_bin_and_out_divs` is documented to return two tuples representing bin divisions and output divisions for resampling operations. These tuples are used together throughout the codebase, with the explicit assumption that they have the same length. For example:

1. Existing tests in the codebase explicitly check `assert len(newdivs) == len(outdivs)`
2. The ResampleAggregation class uses both tuples to create BlockwiseDep objects, assuming they have compatible lengths
3. The divisions are reconstructed using `list(self.divisions_left.iterable) + [self.divisions_right.iterable[-1]]`, which assumes the lengths match

The bug occurs in the "Adjust ends" section (lines 89-102) where the code conditionally appends to `newdivs` but doesn't always append to `outdivs` in the same way, creating a length mismatch.

## Fix

```diff
--- a/resample.py
+++ b/resample.py
@@ -94,10 +94,10 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         if len(newdivs) < len(divs):
             setter = lambda a, val: a.append(val)
         else:
             setter = lambda a, val: a.__setitem__(-1, val)
         setter(newdivs, divisions[-1] + res)
-        if outdivs[-1] > divisions[-1]:
+        if outdivs[-1] >= divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
             setter(outdivs, temp.index[-1])
```

The issue is that when `outdivs[-1] == divisions[-1]`, the code doesn't call `setter` on `outdivs`, but it has already called `setter` on `newdivs`. By changing the condition from `>` to `>=`, we ensure that `setter` is called on `outdivs` in all cases where it was called on `newdivs`, maintaining the length invariant.