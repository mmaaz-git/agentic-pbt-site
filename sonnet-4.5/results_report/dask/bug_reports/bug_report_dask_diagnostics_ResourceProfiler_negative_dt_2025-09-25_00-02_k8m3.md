# Bug Report: dask.diagnostics.ResourceProfiler Crashes with Negative dt

**Target**: `dask.diagnostics.ResourceProfiler`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ResourceProfiler` accepts negative `dt` (sampling interval) values in its constructor but crashes with `ValueError: sleep length must be non-negative` when the background tracker process attempts to sleep.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.diagnostics import ResourceProfiler
from dask.threaded import get
from operator import add
import time


@given(st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False))
def test_resource_profiler_negative_dt(dt):
    dsk = {'x': 1, 'y': (add, 'x', 10)}

    prof = ResourceProfiler(dt=dt)

    with prof:
        result = get(dsk, 'y')
        time.sleep(0.1)

    assert result == 11
```

**Failing input**: `dt=-1` (or any negative value)

## Reproducing the Bug

```python
from dask.diagnostics import ResourceProfiler
from dask.threaded import get
from operator import add
import time

dsk = {'x': 1, 'y': (add, 'x', 10)}

prof = ResourceProfiler(dt=-1)

with prof:
    result = get(dsk, 'y')
    time.sleep(0.2)
```

Output:
```
Process _Tracker-1:
Traceback (most recent call last):
  File "/multiprocessing/process.py", line 313, in _bootstrap
    self.run()
  File "/dask/diagnostics/profile.py", line 286, in run
    sleep(self.dt)
ValueError: sleep length must be non-negative
```

## Why This Is A Bug

The constructor accepts negative `dt` values without validation, but the background `_Tracker` process's `run` method calls `sleep(self.dt)` which raises `ValueError` for negative values. This causes an unhandled exception in the background process.

This violates the principle of fail-fast: invalid parameters should be rejected at construction time, not when they are first used.

## Fix

Add validation in the `__init__` method to reject negative dt values:

```diff
--- a/dask/diagnostics/profile.py
+++ b/dask/diagnostics/profile.py
@@ -161,6 +161,8 @@ class ResourceProfiler(Callback):
     def __init__(self, dt=1):
+        if dt < 0:
+            raise ValueError(f"dt must be non-negative, got {dt}")
         self._dt = dt
         self._entered = False
         self._tracker = None
```