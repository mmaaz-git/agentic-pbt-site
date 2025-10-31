# Bug Report: dask.diagnostics.ProgressBar Crashes with Negative dt

**Target**: `dask.diagnostics.ProgressBar`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ProgressBar` accepts negative `dt` (update interval) values in its constructor but crashes with `ValueError: sleep length must be non-negative` when the background timer thread attempts to sleep.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.diagnostics import ProgressBar
from dask.threaded import get
from operator import add
import io


@given(st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False))
def test_progress_bar_negative_dt(dt):
    output = io.StringIO()
    dsk = {'x': 1, 'y': (add, 'x', 10)}

    with ProgressBar(dt=dt, out=output):
        result = get(dsk, 'y')

    assert result == 11
```

**Failing input**: `dt=-0.1` (or any negative value)

## Reproducing the Bug

```python
from dask.diagnostics import ProgressBar
from dask.threaded import get
from operator import add
import io
import time

output = io.StringIO()
dsk = {'x': 1, 'y': (add, 'x', 10)}

pbar = ProgressBar(dt=-0.1, out=output)

with pbar:
    result = get(dsk, 'y')
    time.sleep(0.2)
```

Output:
```
ValueError: sleep length must be non-negative
  File "/dask/diagnostics/progress.py", line 129, in _timer_func
    time.sleep(self._dt)
```

## Why This Is A Bug

The constructor accepts negative `dt` values without validation, but the background timer thread's `_timer_func` method calls `time.sleep(self._dt)` which raises `ValueError` for negative values. This causes an unhandled exception in the background thread.

This violates the principle of fail-fast: invalid parameters should be rejected at construction time, not when they are first used.

## Fix

Add validation in the `__init__` method to reject negative dt values:

```diff
--- a/dask/diagnostics/progress.py
+++ b/dask/diagnostics/progress.py
@@ -82,6 +82,8 @@ class ProgressBar(Callback):
     def __init__(self, minimum=0, width=40, dt=0.1, out=None):
         if out is None:
             # Warning, on windows, stdout can still be None if
             # an application is started as GUI Application
             # https://docs.python.org/3/library/sys.html#sys.__stderr__
             out = sys.stdout
+        if dt < 0:
+            raise ValueError(f"dt must be non-negative, got {dt}")
         self._minimum = minimum
         self._width = width
```