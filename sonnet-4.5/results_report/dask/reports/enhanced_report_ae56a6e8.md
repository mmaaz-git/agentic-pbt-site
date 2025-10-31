# Bug Report: dask.dataframe.tseries.resample Non-Monotonic Output Divisions

**Target**: `dask.dataframe.tseries.resample._resample_bin_and_out_divs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_resample_bin_and_out_divs` function produces non-monotonic output divisions when `label='right'` and `closed='right'` are used together with weekly resampling rules, causing divisions to go backward in time and violating dask's fundamental requirement of monotonically increasing divisions.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def divisions_strategy(draw):
    n_divs = draw(st.integers(min_value=2, max_value=20))
    start = draw(st.datetimes(
        min_value=pd.Timestamp("2000-01-01"),
        max_value=pd.Timestamp("2020-01-01")
    ))
    freq = draw(st.sampled_from(['1h', '1D', '1min', '30min', '1W']))
    divisions = pd.date_range(start=start, periods=n_divs, freq=freq)
    return tuple(divisions)


@given(
    divisions=divisions_strategy(),
    rule=st.sampled_from(['1h', '2h', '1D', '2D', '1W', '30min', '15min']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=1000)
def test_resample_bin_and_out_divs_monotonic(divisions, rule, closed, label):
    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] <= outdivs[i+1], f"outdivs not monotonic: {outdivs[i]} > {outdivs[i+1]}"


if __name__ == "__main__":
    # Run the test
    test_resample_bin_and_out_divs_monotonic()
```

<details>

<summary>
**Failing input**: `divisions=(Timestamp('2001-01-07 00:00:00'), Timestamp('2001-01-07 01:00:00')), rule='1W', closed='right', label='right'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 34, in <module>
    test_resample_bin_and_out_divs_monotonic()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 19, in test_resample_bin_and_out_divs_monotonic
    divisions=divisions_strategy(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 29, in test_resample_bin_and_out_divs_monotonic
    assert outdivs[i] <= outdivs[i+1], f"outdivs not monotonic: {outdivs[i]} > {outdivs[i+1]}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: outdivs not monotonic: 2001-01-07 00:00:00 > 2000-12-31 00:00:00
Falsifying example: test_resample_bin_and_out_divs_monotonic(
    divisions=(Timestamp('2001-01-07 00:00:00'),
     Timestamp('2001-01-07 01:00:00')),
    rule='1W',
    closed='right',
    label='right',
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Test case from the bug report
divisions = (pd.Timestamp('2001-02-04 00:00:00'), pd.Timestamp('2001-02-04 01:00:00'))
rule = '1W'
closed = 'right'
label = 'right'

print(f"Testing with:")
print(f"  divisions = {divisions}")
print(f"  rule = '{rule}'")
print(f"  closed = '{closed}'")
print(f"  label = '{label}'")
print()

newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

print(f"Result:")
print(f"  newdivs: {newdivs}")
print(f"  outdivs: {outdivs}")
print()

# Check if outdivs is monotonic
is_monotonic = all(outdivs[i] <= outdivs[i+1] for i in range(len(outdivs)-1))
print(f"Monotonicity check:")
print(f"  Are outdivs monotonic? {is_monotonic}")

if not is_monotonic:
    print(f"\nERROR: Non-monotonic divisions detected!")
    for i in range(len(outdivs)-1):
        if outdivs[i] > outdivs[i+1]:
            print(f"  outdivs[{i}] = {outdivs[i]} > outdivs[{i+1}] = {outdivs[i+1]}")
```

<details>

<summary>
Non-monotonic divisions detected: 2001-02-04 > 2001-01-28
</summary>
```
Testing with:
  divisions = (Timestamp('2001-02-04 00:00:00'), Timestamp('2001-02-04 01:00:00'))
  rule = '1W'
  closed = 'right'
  label = 'right'

Result:
  newdivs: (Timestamp('2001-02-04 00:00:00'), Timestamp('2001-02-05 01:00:00'))
  outdivs: (Timestamp('2001-02-04 00:00:00'), Timestamp('2001-01-28 00:00:00'))

Monotonicity check:
  Are outdivs monotonic? False

ERROR: Non-monotonic divisions detected!
  outdivs[0] = 2001-02-04 00:00:00 > outdivs[1] = 2001-01-28 00:00:00
```
</details>

## Why This Is A Bug

This bug violates a fundamental requirement of dask's architecture: **divisions must be monotonically increasing**. The function produces output divisions where `outdivs[0] = 2001-02-04` comes *after* `outdivs[1] = 2001-01-28`, creating a backward time jump of one week.

This violation occurs specifically when:
1. Both `closed='right'` and `label='right'` are set
2. The resampling rule is weekly ('1W', '2W', etc.)
3. The division span is shorter than the resampling period

The non-monotonic divisions break dask's data partitioning system, which relies on ordered divisions to correctly distribute and locate data across partitions. This would lead to:
- Incorrect data partitioning where data ends up in wrong partitions
- Query failures when searching for data within time ranges
- Potential data loss or corruption in aggregation operations
- Runtime errors in downstream operations that assume monotonic divisions

## Relevant Context

The bug occurs in the `_resample_bin_and_out_divs` function at line 66-103 of `/dask/dataframe/tseries/resample.py`. This internal function is used by the public resampling API to determine how to partition data for time-based resampling operations.

The function handles two transformations:
1. When `label='right'` (line 81-82), it shifts `outdivs` forward by the rule amount: `outdivs = tempdivs + rule`
2. During end adjustment (lines 98-101), when the last division needs to be extended, it appends `temp.index[-1]`

The bug manifests because `temp.index[-1]` is the unshifted value from the original resampling, but it gets appended to the shifted `outdivs` array. This creates the non-monotonic sequence when `label='right'` because the appended value is one week behind the already-shifted first value.

Documentation: https://docs.dask.org/en/stable/dataframe-api.html#dask.dataframe.DataFrame.resample
Code location: https://github.com/dask/dask/blob/main/dask/dataframe/tseries/resample.py#L66-L103

## Proposed Fix

```diff
--- a/dask/dataframe/tseries/resample.py
+++ b/dask/dataframe/tseries/resample.py
@@ -98,7 +98,10 @@ def _resample_bin_and_out_divs(divisions, rule, closed="left", label="left"):
         if outdivs[-1] > divisions[-1]:
             setter(outdivs, outdivs[-1])
         elif outdivs[-1] < divisions[-1]:
-            setter(outdivs, temp.index[-1])
+            if g.label == "right":
+                setter(outdivs, temp.index[-1] + rule)
+            else:
+                setter(outdivs, temp.index[-1])

     return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))
```