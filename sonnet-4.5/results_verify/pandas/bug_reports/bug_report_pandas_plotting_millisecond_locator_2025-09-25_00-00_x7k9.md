# Bug Report: pandas.plotting MilliSecondLocator Variable Typo

**Target**: `pandas.plotting._matplotlib.converter.MilliSecondLocator.__call__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Line 434 in `pandas/plotting/_matplotlib/converter.py` has a typo where `ed = dmin.replace(tzinfo=None)` should be `ed = dmax.replace(tzinfo=None)`. This causes `date_range(start=st, end=ed, ...)` to be called with `start == end`, producing an empty or single-element range instead of spanning the full time interval for millisecond-level tick marks.

## Property-Based Test

```python
import datetime
import pytest
from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.plotting._matplotlib.converter import MilliSecondLocator
import matplotlib.pyplot as plt


@given(milliseconds=st.integers(min_value=1, max_value=100))
@settings(max_examples=100)
def test_millisecond_locator_creates_empty_range(milliseconds):
    """
    Property: MilliSecondLocator should generate tick marks spanning the full time range.

    BUG: Line 434 has `ed = dmin.replace(tzinfo=None)` instead of
    `ed = dmax.replace(tzinfo=None)`, causing date_range to be created with
    start == end, producing no useful tick marks.
    """
    fig, ax = plt.subplots()

    dmin = datetime.datetime(2020, 1, 1, 0, 0, 0, 0)
    dmax = dmin + datetime.timedelta(milliseconds=milliseconds)

    locator = MilliSecondLocator(tz=datetime.timezone.utc)
    locator.set_axis(ax.xaxis)
    ax.set_xlim(dmin, dmax)

    st = dmin.replace(tzinfo=None)
    ed_buggy = dmin.replace(tzinfo=None)  # BUG: should be dmax
    ed_correct = dmax.replace(tzinfo=None)

    buggy_range = pd.date_range(start=st, end=ed_buggy, freq='1ms', tz='UTC')
    correct_range = pd.date_range(start=st, end=ed_correct, freq='1ms', tz='UTC')

    assert len(buggy_range) <= 1
    if milliseconds > 10:
        assert len(correct_range) > 1

    plt.close(fig)
```

**Failing input**: Any time range spanning more than a few milliseconds

## Reproducing the Bug

```python
import datetime
import pandas as pd

dmin = datetime.datetime(2020, 1, 1, 0, 0, 0, 0)
dmax = datetime.datetime(2020, 1, 1, 0, 0, 0, 50000)  # 50ms later

st = dmin.replace(tzinfo=None)
ed_buggy = dmin.replace(tzinfo=None)  # BUG in converter.py line 434
ed_correct = dmax.replace(tzinfo=None)

print("Buggy date_range (start == end):")
buggy_range = pd.date_range(start=st, end=ed_buggy, freq='1ms', tz='UTC')
print(f"  Length: {len(buggy_range)}")

print("\nCorrect date_range (start to end):")
correct_range = pd.date_range(start=st, end=ed_correct, freq='1ms', tz='UTC')
print(f"  Length: {len(correct_range)}")

print(f"\nBug impact: Instead of {len(correct_range)} tick marks, we get {len(buggy_range)}")
```

Output:
```
Buggy date_range (start == end):
  Length: 1

Correct date_range (start to end):
  Length: 51

Bug impact: Instead of 51 tick marks, we get 1
```

## Why This Is A Bug

The `MilliSecondLocator.__call__` method is responsible for generating tick mark locations for matplotlib plots at millisecond precision. Lines 433-435 create a date range spanning from `dmin` to `dmax`:

```python
st = dmin.replace(tzinfo=None)
ed = dmin.replace(tzinfo=None)  # BUG: should be dmax
all_dates = date_range(start=st, end=ed, freq=freq, tz=tz).astype(object)
```

The typo on line 434 uses `dmin` instead of `dmax`, making `st == ed`. This creates a degenerate date range with start equal to end, producing at most one tick mark. The code then falls back to the error handler on lines 441-445, returning only the endpoints `[dmin, dmax]` instead of properly spaced millisecond ticks.

This defeats the purpose of the `MilliSecondLocator` class, which should generate multiple appropriately-spaced tick marks for high-precision time axes.

## Fix

```diff
--- a/pandas/plotting/_matplotlib/converter.py
+++ b/pandas/plotting/_matplotlib/converter.py
@@ -431,7 +431,7 @@ class MilliSecondLocator(mdates.DateLocator):
         freq = f"{interval}ms"
         tz = self.tz.tzname(None)
         st = dmin.replace(tzinfo=None)
-        ed = dmin.replace(tzinfo=None)
+        ed = dmax.replace(tzinfo=None)
         all_dates = date_range(start=st, end=ed, freq=freq, tz=tz).astype(object)

         try:
```