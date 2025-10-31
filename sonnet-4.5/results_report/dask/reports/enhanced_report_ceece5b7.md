# Bug Report: dask.dataframe.tseries Produces Duplicate Division Timestamps

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function generates duplicate consecutive timestamps in output divisions when the resample frequency exceeds the input time range, violating Dask's strict monotonicity requirement for DataFrame divisions.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

@st.composite
def timestamp_list_strategy(draw):
    size = draw(st.integers(min_value=2, max_value=20))
    start = draw(st.datetimes(
        min_value=pd.Timestamp('2000-01-01'),
        max_value=pd.Timestamp('2020-01-01')
    ))
    freq_hours = draw(st.integers(min_value=1, max_value=24*7))
    timestamps = pd.date_range(start=start, periods=size, freq=f'{freq_hours}h')
    return timestamps.tolist()

@given(
    divisions=timestamp_list_strategy(),
    rule=st.sampled_from(['1h', '2h', '6h', '12h', '1D', '2D']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=500, deadline=None)
def test_no_duplicate_divisions(divisions, rule, closed, label):
    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] < outdivs[i+1], \
            f"Divisions not strictly increasing: outdivs[{i}]={outdivs[i]}, outdivs[{i+1}]={outdivs[i+1]}"

if __name__ == "__main__":
    # Run the test to find failing cases
    test_no_duplicate_divisions()
```

<details>

<summary>
**Failing input**: `divisions=[Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 01:00:00')], rule='2h', closed='left', label='left'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 32, in <module>
    test_no_duplicate_divisions()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 17, in test_no_duplicate_divisions
    divisions=timestamp_list_strategy(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 27, in test_no_duplicate_divisions
    assert outdivs[i] < outdivs[i+1], \
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Divisions not strictly increasing: outdivs[0]=2000-01-01 00:00:00, outdivs[1]=2000-01-01 00:00:00
Falsifying example: test_no_duplicate_divisions(
    divisions=[Timestamp('2000-01-01 00:00:00'),
     Timestamp('2000-01-01 01:00:00')],
    rule='2h',
    closed='left',
    label='left',  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/0/hypo.py:28
        /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py:94
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Minimal test case showing the bug
divisions = [pd.Timestamp('2000-01-01 00:00:00'), pd.Timestamp('2000-01-01 01:00:00')]
rule = '2h'
closed = 'left'
label = 'left'

print("Input:")
print(f"  divisions: {divisions}")
print(f"  rule: '{rule}'")
print(f"  closed: '{closed}'")
print(f"  label: '{label}'")
print()

# Call the function
newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

print("Output:")
print(f"  newdivs: {newdivs}")
print(f"  outdivs: {outdivs}")
print()

# Check for duplicates in outdivs
print("Checking for duplicates in outdivs:")
for i in range(len(outdivs) - 1):
    if outdivs[i] >= outdivs[i+1]:
        print(f"  ERROR: outdivs[{i}] ({outdivs[i]}) >= outdivs[{i+1}] ({outdivs[i+1]})")
        print(f"  Divisions are not strictly monotonic increasing!")
    else:
        print(f"  OK: outdivs[{i}] < outdivs[{i+1}]")

if len(outdivs) > 1 and outdivs[0] == outdivs[-1]:
    print()
    print("CRITICAL BUG: All division values are identical!")
    print("This violates Dask's fundamental requirement that divisions be strictly monotonic.")
```

<details>

<summary>
Output showing duplicate division timestamps
</summary>
```
Input:
  divisions: [Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 01:00:00')]
  rule: '2h'
  closed: 'left'
  label: 'left'

Output:
  newdivs: (Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 01:00:00.000000001'))
  outdivs: (Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 00:00:00'))

Checking for duplicates in outdivs:
  ERROR: outdivs[0] (2000-01-01 00:00:00) >= outdivs[1] (2000-01-01 00:00:00)
  Divisions are not strictly monotonic increasing!

CRITICAL BUG: All division values are identical!
This violates Dask's fundamental requirement that divisions be strictly monotonic.
```
</details>

## Why This Is A Bug

Dask DataFrames require divisions to be strictly monotonically increasing values that define partition boundaries. This is a fundamental architectural requirement documented throughout the codebase. When `_resample_bin_and_out_divs` produces duplicate consecutive timestamps, it violates this invariant in the following ways:

1. **Partition Ambiguity**: With duplicate division values, Dask cannot determine which partition should contain data with timestamps between the duplicated values. This creates zero-width partitions that break the partitioning logic.

2. **Index Operations Fail**: Operations like `loc`, `merge`, and `groupby` rely on monotonic divisions to efficiently locate data. The code in `dask/dataframe/methods.py:108` explicitly checks `is_monotonic_increasing` and handles monotonic vs non-monotonic indexes differently.

3. **Silent Data Corruption**: Since many Dask operations assume monotonic divisions without runtime checks, duplicate divisions can lead to incorrect results being returned silently.

4. **Contract Violation**: The function's output is used directly by `ResampleReduction._resample_divisions` (line 147-149) to set divisions for downstream operations, propagating the invalid state throughout the computation graph.

The bug occurs when the resample frequency (`rule='2h'`) is larger than or equal to the input time range (1 hour between divisions). In this case, the pandas resampling creates a single bin, but the end-adjustment logic (lines 89-101) incorrectly modifies `outdivs` by setting `outdivs[-1]` to `temp.index[-1]` (line 101), which equals `outdivs[0]`, creating the duplicate.

## Relevant Context

The `_resample_bin_and_out_divs` function is an internal helper used by the public resample API through the `ResampleReduction` class hierarchy. While it's not directly exposed to users, it's critical for the correctness of all time series resampling operations in Dask.

Key code locations:
- Bug location: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/tseries/resample.py:101`
- Called by: `ResampleReduction._resample_divisions` (line 147-149)
- Monotonicity checks: `dask/dataframe/methods.py:108` checks `is_monotonic_increasing`

The issue manifests specifically when:
1. Input time range < resample frequency (e.g., 1 hour of data resampled to 2-hour bins)
2. The condition `outdivs[-1] < divisions[-1]` is true (line 100)
3. The setter uses `temp.index[-1]` which may equal an earlier division value

## Proposed Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -98,7 +98,13 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            new_val = temp.index[-1]
+            # Ensure strict monotonicity - if the new value would create
+            # a duplicate or non-monotonic sequence, adjust it
+            if len(outdivs) > 1 and new_val <= outdivs[-2]:
+                # Add the rule frequency to maintain monotonicity
+                new_val = outdivs[-2] + rule
+            setter(outdivs, new_val)

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```