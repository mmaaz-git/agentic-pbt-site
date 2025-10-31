# Bug Report: dask.diagnostics.CacheProfiler Empty metric_name Ignored

**Target**: `dask.diagnostics.CacheProfiler.__init__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`CacheProfiler` ignores explicitly provided empty string `metric_name=""` parameter due to truthiness check instead of `is not None` check, treating it the same as `None`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.diagnostics import CacheProfiler


def custom_metric(value):
    return len(str(value))


@given(st.text())
def test_cache_profiler_custom_metric_name(metric_name):
    prof = CacheProfiler(metric=custom_metric, metric_name=metric_name)
    assert prof._metric_name == metric_name
```

**Failing input**: `metric_name=''`

## Reproducing the Bug

```python
from dask.diagnostics import CacheProfiler


def custom_metric(value):
    return len(str(value))


prof = CacheProfiler(metric=custom_metric, metric_name="")

print(f"Expected: metric_name = ''")
print(f"Actual:   metric_name = '{prof._metric_name}'")

assert prof._metric_name == ""
```

Output:
```
Expected: metric_name = ''
Actual:   metric_name = 'custom_metric'
AssertionError
```

## Why This Is A Bug

The code uses a truthiness check (`if metric_name:`) on line 353 of `dask/diagnostics/profile.py`, which treats empty string as falsy. This causes the explicitly provided `metric_name=""` to be ignored and fall through to using `metric.__name__` instead.

This violates the principle of least surprise because:
1. Empty string is a valid string value that a user might intentionally provide
2. The code should distinguish between "not provided" (`None`) and "provided as empty" (`""`)
3. The behavior is inconsistent: whitespace-only strings like `"   "` are accepted, but `""` is not

## Fix

```diff
--- a/dask/diagnostics/profile.py
+++ b/dask/diagnostics/profile.py
@@ -350,7 +350,7 @@ class CacheProfiler(Callback):
     def __init__(self, metric=None, metric_name=None):
         self.clear()
         self._metric = metric if metric else lambda value: 1
-        if metric_name:
+        if metric_name is not None:
             self._metric_name = metric_name
         elif metric:
             self._metric_name = metric.__name__
```