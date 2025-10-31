# Bug Report: dask.dataframe.tseries.resample._resample_bin_and_out_divs Returns Mismatched Division Lengths

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function returns `newdivs` and `outdivs` tuples with different lengths when called with `closed='right'` and `label='right'`, breaking the downstream `ResampleAggregation` class which requires equal-length division arrays.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

@given(
    st.integers(min_value=3, max_value=50),
    st.sampled_from(['h', 'D', '2h', '6h', '12h']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right'])
)
@settings(max_examples=1000)
def test_resample_divisions_with_gaps(n_segments, rule, closed, label):
    segments = []
    start_dates = ['2020-01-01', '2020-02-01']

    for start_str in start_dates:
        start = pd.Timestamp(start_str)
        end = start + pd.Timedelta(days=7)
        segment = pd.date_range(start, end, periods=n_segments // 2)
        segments.append(segment)

    divisions = segments[0].union(segments[1])

    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

    assert len(newdivs) == len(outdivs), f"Length mismatch: newdivs={len(newdivs)}, outdivs={len(outdivs)}"

if __name__ == "__main__":
    test_resample_divisions_with_gaps()
```

<details>

<summary>
**Failing input**: `n_segments=18, rule='D', closed='right', label='right'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 29, in <module>
    test_resample_divisions_with_gaps()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 6, in test_resample_divisions_with_gaps
    st.integers(min_value=3, max_value=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 26, in test_resample_divisions_with_gaps
    assert len(newdivs) == len(outdivs), f"Length mismatch: newdivs={len(newdivs)}, outdivs={len(outdivs)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Length mismatch: newdivs=17, outdivs=16
Falsifying example: test_resample_divisions_with_gaps(
    n_segments=18,
    rule='D',
    closed='right',
    label='right',
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

segments = []
for start_str in ['2020-01-01', '2020-02-01']:
    start = pd.Timestamp(start_str)
    end = start + pd.Timedelta(days=7)
    segment = pd.date_range(start, end, periods=9)
    segments.append(segment)

divisions = segments[0].union(segments[1])

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'D', closed='right', label='right')

print(f"len(newdivs) = {len(newdivs)}")
print(f"len(outdivs) = {len(outdivs)}")
print(f"Length mismatch: {len(newdivs) != len(outdivs)}")
print(f"\nnewdivs: {newdivs}")
print(f"\noutdivs: {outdivs}")
```

<details>

<summary>
Length mismatch detected: newdivs has 17 elements, outdivs has 16 elements
</summary>
```
len(newdivs) = 17
len(outdivs) = 16
Length mismatch: True

newdivs: (Timestamp('2020-01-01 00:00:00'), Timestamp('2020-01-01 00:00:00.000000001'), Timestamp('2020-01-02 00:00:00.000000001'), Timestamp('2020-01-03 00:00:00.000000001'), Timestamp('2020-01-04 00:00:00.000000001'), Timestamp('2020-01-05 00:00:00.000000001'), Timestamp('2020-01-06 00:00:00.000000001'), Timestamp('2020-01-07 00:00:00.000000001'), Timestamp('2020-01-31 00:00:00.000000001'), Timestamp('2020-02-01 00:00:00.000000001'), Timestamp('2020-02-02 00:00:00.000000001'), Timestamp('2020-02-03 00:00:00.000000001'), Timestamp('2020-02-04 00:00:00.000000001'), Timestamp('2020-02-05 00:00:00.000000001'), Timestamp('2020-02-06 00:00:00.000000001'), Timestamp('2020-02-07 00:00:00.000000001'), Timestamp('2020-02-08 00:00:00.000000001'))

outdivs: (Timestamp('2020-01-01 00:00:00'), Timestamp('2020-01-02 00:00:00'), Timestamp('2020-01-03 00:00:00'), Timestamp('2020-01-04 00:00:00'), Timestamp('2020-01-05 00:00:00'), Timestamp('2020-01-06 00:00:00'), Timestamp('2020-01-07 00:00:00'), Timestamp('2020-01-08 00:00:00'), Timestamp('2020-02-01 00:00:00'), Timestamp('2020-02-02 00:00:00'), Timestamp('2020-02-03 00:00:00'), Timestamp('2020-02-04 00:00:00'), Timestamp('2020-02-05 00:00:00'), Timestamp('2020-02-06 00:00:00'), Timestamp('2020-02-07 00:00:00'), Timestamp('2020-02-08 00:00:00'))
```
</details>

## Why This Is A Bug

This violates an implicit contract between `_resample_bin_and_out_divs` and its consumer `ResampleAggregation`. The `ResampleAggregation` class expects `divisions_left` and `divisions_right` to have equal lengths:

1. In `ResampleReduction._lower()` (lines 160-164), it creates `BlockwiseDep` objects from the output divisions assuming parallel structure
2. In `ResampleAggregation._divisions()` (line 194), it constructs the final divisions by concatenating all elements from `divisions_left` with the last element of `divisions_right`
3. The blockwise operations in `ResampleAggregation._blockwise_arg()` (lines 196-199) iterate over these divisions assuming equal indexing

When lengths differ, the blockwise operations fail with index out of bounds errors or produce corrupted data. The bug occurs specifically when both `closed='right'` and `label='right'` are set, triggering a code path in lines 90-102 where `newdivs` gets updated via the setter but `outdivs` does not when `outdivs[-1] == divisions[-1]`.

## Relevant Context

The bug manifests in the adjustment logic at the end of `_resample_bin_and_out_divs` (lines 90-102 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py`):

- When `newdivs[-1] < divisions[-1]`, the code determines whether to append or update the last element
- A setter function is created based on whether the array needs to grow
- The setter is always called on `newdivs` (line 97)
- The setter is conditionally called on `outdivs` based on its relationship to `divisions[-1]`
- When `outdivs[-1] == divisions[-1]`, no setter is called on `outdivs`, creating the length mismatch

This is a logic error where one branch (the equality case) fails to maintain the parallel structure between the two arrays.

## Proposed Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -95,9 +95,11 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         else:
             setter = lambda a, val: a.__setitem__(-1, val)
         setter(newdivs, divisions[-1] + res)
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
             setter(outdivs, temp.index[-1])
+        else:  # outdivs[-1] == divisions[-1]
+            setter(outdivs, outdivs[-1])

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```