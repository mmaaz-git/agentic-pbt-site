# Bug Report: dask.diagnostics.CacheProfiler KeyError with Partial Task Graphs

**Target**: `dask.diagnostics.profile.CacheProfiler`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`CacheProfiler._posttask` and `CacheProfiler._finish` crash with a `KeyError` when accessing task definitions from a partial task graph that doesn't contain all cached tasks.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.diagnostics import CacheProfiler


@given(st.text(min_size=1), st.text(min_size=1))
def test_cache_profiler_handles_partial_graphs(key1, key2):
    prof = CacheProfiler()

    dsk1 = {key1: 1}
    prof._start(dsk1)

    state1 = {"released": set()}
    prof._posttask(key1, 1, dsk1, state1, 1)

    dsk2 = {key2: 2}
    state2 = {"released": {key1}}

    prof._posttask(key2, 2, dsk2, state2, 1)
```

**Failing input**: Any two distinct keys, e.g., `key1="x"`, `key2="y"`

## Reproducing the Bug

```python
from dask.diagnostics import CacheProfiler

prof = CacheProfiler()

dsk1 = {"x": 1, "y": ("add", "x", 10)}
prof._start(dsk1)

state1 = {"released": set()}
prof._posttask("y", 11, dsk1, state1, 1)

dsk2 = {"z": ("mul", "y", 2)}
state2 = {"released": {"y"}}

prof._posttask("z", 22, dsk2, state2, 1)
```

Output:
```
KeyError: 'y'
```

## Why This Is A Bug

The `CacheProfiler` class accumulates task definitions in `self._dsk` via the `_start` method. However, in `_posttask` (line 377) and `_finish` (line 382), it accesses tasks using `dsk[k]` instead of `self._dsk[k]`, where `dsk` is the method parameter.

When the dask scheduler passes a partial task graph to these methods, tasks that were cached from previous graphs but aren't in the current partial graph cause a `KeyError`.

This is inconsistent with the profiler's design - `self._dsk` is maintained precisely to have a complete record of all tasks across multiple graph executions.

## Fix

```diff
--- a/dask/diagnostics/profile.py
+++ b/dask/diagnostics/profile.py
@@ -374,7 +374,7 @@ class CacheProfiler(Callback):
         self._cache[key] = (self._metric(value), t)
         for k in state["released"] & self._cache.keys():
             metric, start = self._cache.pop(k)
-            self.results.append(CacheData(k, dsk[k], metric, start, t))
+            self.results.append(CacheData(k, self._dsk[k], metric, start, t))

     def _finish(self, dsk, state, failed):
         t = default_timer()
@@ -379,7 +379,7 @@ class CacheProfiler(Callback):
     def _finish(self, dsk, state, failed):
         t = default_timer()
         for k, (metric, start) in self._cache.items():
-            self.results.append(CacheData(k, dsk[k], metric, start, t))
+            self.results.append(CacheData(k, self._dsk[k], metric, start, t))
         self._cache.clear()
```