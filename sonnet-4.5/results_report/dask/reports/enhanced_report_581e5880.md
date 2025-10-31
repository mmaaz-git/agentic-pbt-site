# Bug Report: dask.dataframe.dask_expr._rolling TypeError with Timedelta Windows and center=True

**Target**: `dask.dataframe.dask_expr._rolling.RollingReduction._lower`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Dask DataFrame crashes with a `TypeError` when using rolling windows with string/timedelta specifications (e.g., '2h', '1D') combined with `center=True` on datetime-indexed DataFrames, attempting integer division on a string value.

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

<details>

<summary>
**Failing input**: `window_value='1h', center=True, npartitions=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 35, in <module>
    test_rolling_window_type_compatibility()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 8, in test_rolling_window_type_compatibility
    window_value=st.sampled_from(['1h', '2h', '1D', '30min']),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 25, in test_rolling_window_type_compatibility
    dask_computed = dask_result.compute()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    (result,) = compute(self, traverse=False, **kwargs)
                ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 678, in compute
    expr = expr.optimize()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 440, in optimize
    return optimize_until(self, stage)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 944, in optimize_until
    expr = expr.lower_completely()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 523, in lower_completely
    new = expr.lower_once(lowered)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 489, in lower_once
    new = operand.lower_once(lowered)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 489, in lower_once
    new = operand.lower_once(lowered)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 478, in lower_once
    out = expr._lower()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_rolling.py", line 128, in _lower
    before = self.window // 2
             ~~~~~~~~~~~~^^~~
TypeError: unsupported operand type(s) for //: 'str' and 'int'
Falsifying example: test_rolling_window_type_compatibility(
    window_value='1h',  # or any other generated value
    center=True,
    npartitions=2,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_rolling.py:128
        /home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py:2272
        /home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py:2277
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe import from_pandas

# Create a simple DataFrame with datetime index
df = pd.DataFrame({
    'time': pd.date_range('2020-01-01', periods=10, freq='1h'),
    'value': range(10)
})
df = df.set_index('time')

# Convert to Dask DataFrame
ddf = from_pandas(df, npartitions=2)

# This will crash with TypeError
try:
    result = ddf.rolling(window='2h', center=True).mean()
    computed = result.compute()
    print("Success! Result shape:", computed.shape)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TypeError when computing rolling window with string window and center=True
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/repo.py", line 17, in <module>
    computed = result.compute()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    (result,) = compute(self, traverse=False, **kwargs)
                ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 678, in compute
    expr = expr.optimize()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 440, in optimize
    return optimize_until(self, stage)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 944, in optimize_until
    expr = expr.lower_completely()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 523, in lower_completely
    new = expr.lower_once(lowered)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 489, in lower_once
    new = operand.lower_once(lowered)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 489, in lower_once
    new = operand.lower_once(lowered)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/_expr.py", line 478, in lower_once
    out = expr._lower()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_rolling.py", line 128, in _lower
    before = self.window // 2
             ~~~~~~~~~~~~^^~~
TypeError: unsupported operand type(s) for //: 'str' and 'int'
Error: TypeError: unsupported operand type(s) for //: 'str' and 'int'
```
</details>

## Why This Is A Bug

This violates expected behavior because:
1. **Pandas compatibility**: Pandas successfully handles `df.rolling(window='2h', center=True).mean()` on datetime-indexed DataFrames, computing centered time-based rolling windows correctly
2. **Dask API promise**: Dask explicitly claims to provide a pandas-compatible DataFrame API, and both window strings and center parameters are documented features
3. **Partial implementation**: The same operation works in Dask when `center=False`, indicating the feature is intended to work but has an implementation oversight
4. **Type assumption error**: The code at line 128 in `_rolling.py` incorrectly assumes `self.window` is always an integer when `center=True`, performing `self.window // 2` without type checking
5. **Documentation gap**: Neither Dask nor pandas documentation indicates that combining string/timedelta windows with `center=True` should be unsupported

## Relevant Context

The bug occurs in `/dask/dataframe/dask_expr/_rolling.py` at lines 127-129 in the `_lower` method of the `RollingReduction` class. The code logic branches based on whether centering is enabled but fails to account for non-integer window types when calculating the before/after overlap values for centering.

Key observations:
- Works correctly when `center=False` with time-based windows (lines 130-132 handle this case properly)
- Works correctly when `center=True` with integer windows
- Only fails with the combination of time-based windows AND `center=True`
- The issue affects all aggregation operations (mean, sum, max, etc.) since they all inherit from `RollingReduction`

Relevant code location: https://github.com/dask/dask-expr/blob/main/dask_expr/_rolling.py

## Proposed Fix

```diff
--- a/dask/dataframe/dask_expr/_rolling.py
+++ b/dask/dataframe/dask_expr/_rolling.py
@@ -125,8 +125,13 @@ class RollingReduction(Expr):
             )

         if self.kwargs.get("center"):
-            before = self.window // 2
-            after = self.window - before - 1
+            if isinstance(self.window, int):
+                before = self.window // 2
+                after = self.window - before - 1
+            else:
+                # For time-based windows, split the timedelta in half
+                td = pd.Timedelta(self.window)
+                before = after = td / 2
         elif not isinstance(self.window, int):
             before = pd.Timedelta(self.window)
             after = 0
```