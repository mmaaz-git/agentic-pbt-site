# Bug Report: dask.diagnostics.ResourceProfiler Accepts Invalid Negative dt Values Leading to Background Process Crash

**Target**: `dask.diagnostics.ResourceProfiler`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ResourceProfiler accepts negative `dt` (sampling interval) values without validation, causing the background _Tracker process to crash with either `ValueError: sleep length must be non-negative` or `OverflowError: timestamp out of range` when it attempts to sleep.

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


# Run the test
if __name__ == "__main__":
    try:
        test_resource_profiler_negative_dt()
        print("Test passed (unexpectedly)")
    except Exception as e:
        print(f"Test failed with: {e}")
        # Run with a specific failing input to show error
        print("\nReproducing with dt=-1:")
        test_resource_profiler_negative_dt(-1)
```

<details>

<summary>
**Failing input**: `dt=-1.0` (or any negative value)
</summary>
```
Process _Tracker-1:
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/multiprocessing/process.py", line 313, in _bootstrap
    self.run()
    ~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/profile.py", line 286, in run
    sleep(self.dt)
    ~~~~~^^^^^^^^^
OverflowError: timestamp out of range for platform time_t
Process _Tracker-2:
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/multiprocessing/process.py", line 313, in _bootstrap
    self.run()
    ~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/profile.py", line 286, in run
    sleep(self.dt)
    ~~~~~^^^^^^^^^
OverflowError: timestamp out of range for platform time_t
Process _Tracker-3:
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/multiprocessing/process.py", line 313, in _bootstrap
    self.run()
    ~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/profile.py", line 286, in run
    sleep(self.dt)
    ~~~~~^^^^^^^^^
ValueError: sleep length must be non-negative
[... Multiple similar errors from Hypothesis testing various negative values...]
```
</details>

## Reproducing the Bug

```python
from dask.diagnostics import ResourceProfiler
from dask.threaded import get
from operator import add
import time

# Create a simple dask graph
dsk = {'x': 1, 'y': (add, 'x', 10)}

# Create ResourceProfiler with negative dt value
prof = ResourceProfiler(dt=-1)

# Use the profiler in a context manager
with prof:
    result = get(dsk, 'y')
    time.sleep(0.2)

print(f"Result: {result}")
```

<details>

<summary>
Process crashes with ValueError in background tracker
</summary>
```
Process _Tracker-1:
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/multiprocessing/process.py", line 313, in _bootstrap
    self.run()
    ~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/profile.py", line 286, in run
    sleep(self.dt)
    ~~~~~^^^^^^^^^
ValueError: sleep length must be non-negative
Result: 11
```
</details>

## Why This Is A Bug

This violates fundamental API design principles:
1. **Fail-fast principle**: Invalid parameters should be rejected at the API boundary (constructor), not when they are first used deep in a background process
2. **Delayed failure**: The error only manifests when the background _Tracker process starts and attempts `sleep(self.dt)`, making debugging harder
3. **Semantic violation**: A negative sampling interval (`dt`) is semantically meaningless - you cannot sample at negative time intervals
4. **Inconsistent error types**: Different negative values cause different exceptions (ValueError vs OverflowError), leading to unpredictable behavior
5. **Poor user experience**: Users see cryptic multiprocessing errors from a background process rather than a clear validation error

While the main computation still completes (returns 11), the resource profiling functionality completely fails, defeating the purpose of using ResourceProfiler.

## Relevant Context

The issue occurs in the ResourceProfiler class at `/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/profile.py`:

- Line 161: `__init__` accepts `dt` without any validation: `self._dt = dt`
- Line 174: A background _Tracker process is created with this dt value
- Line 286: The _Tracker.run() method calls `sleep(self.dt)` which fails for negative values

The documentation and examples only show positive dt values (e.g., `ResourceProfiler(dt=0.5)`), but there's no explicit constraint documented or enforced that dt must be positive.

Python's `time.sleep()` function requires non-negative arguments as per its documentation, which ResourceProfiler violates by passing unchecked user input.

## Proposed Fix

Add validation in the `__init__` method to reject negative dt values immediately:

```diff
--- a/dask/diagnostics/profile.py
+++ b/dask/diagnostics/profile.py
@@ -159,6 +159,8 @@ class ResourceProfiler(Callback):
     """

     def __init__(self, dt=1):
+        if dt < 0:
+            raise ValueError(f"dt must be non-negative, got {dt}")
         self._dt = dt
         self._entered = False
         self._tracker = None
```

This ensures invalid inputs are rejected immediately with a clear error message, rather than causing obscure crashes in background processes.