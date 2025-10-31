# Bug Report: dask.diagnostics.ProgressBar Crashes with Negative Width

**Target**: `dask.diagnostics.ProgressBar`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ProgressBar` accepts negative `width` values in its constructor but crashes with `ValueError: Sign not allowed in string format specifier` when the progress bar is actually displayed.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.diagnostics import ProgressBar
from dask.threaded import get
from operator import add
import io


@given(st.integers(max_value=-1))
def test_progress_bar_negative_width(width):
    output = io.StringIO()
    dsk = {'x': 1, 'y': (add, 'x', 10)}

    with ProgressBar(width=width, out=output):
        result = get(dsk, 'y')

    assert result == 11
```

**Failing input**: `width=-1` (or any negative integer)

## Reproducing the Bug

```python
from dask.diagnostics import ProgressBar
from dask.threaded import get
from operator import add
import io

output = io.StringIO()
dsk = {'x': 1, 'y': (add, 'x', 10)}

pbar = ProgressBar(width=-1, out=output)

with pbar:
    result = get(dsk, 'y')
```

Output:
```
ValueError: Sign not allowed in string format specifier
  File "/dask/diagnostics/progress.py", line 147, in _draw_bar
    msg = "\r[{0:<{1}}] | {2}% Completed | {3}".format(
        bar, self._width, percent, elapsed
    )
```

## Why This Is A Bug

The constructor accepts negative `width` values without validation, but the `_draw_bar` method uses `width` in a Python format specifier `{0:<{1}}` which does not allow negative alignment widths. This causes a crash when the progress bar attempts to display.

This violates the principle of fail-fast: invalid parameters should be rejected at construction time, not when they are first used.

## Fix

Add validation in the `__init__` method to reject non-positive width values:

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
+        if width <= 0:
+            raise ValueError(f"width must be positive, got {width}")
         self._minimum = minimum
         self._width = width
```