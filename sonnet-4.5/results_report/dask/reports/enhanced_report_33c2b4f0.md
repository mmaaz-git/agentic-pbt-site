# Bug Report: dask.diagnostics.ProgressBar Crashes with Negative Width Values

**Target**: `dask.diagnostics.ProgressBar`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ProgressBar's constructor accepts negative `width` values without validation, but crashes with `ValueError: Sign not allowed in string format specifier` when the progress bar tries to render.

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

if __name__ == "__main__":
    test_progress_bar_negative_width()
```

<details>

<summary>
**Failing input**: `width=-1` (or any negative integer)
</summary>
```
Exception in thread Thread-1 (_timer_func):
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/threading.py", line 1041, in _bootstrap_inner
    self.run()
    ~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/threading.py", line 992, in run
    self._target(*self._args, **self._kwargs)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/progress.py", line 128, in _timer_func
    self._update_bar(elapsed)
    ~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/progress.py", line 134, in _update_bar
    self._draw_bar(0, elapsed)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/progress.py", line 147, in _draw_bar
    msg = "\r[{0:<{1}}] | {2}% Completed | {3}".format(
        bar, self._width, percent, elapsed
    )
ValueError: Sign not allowed in string format specifier
[... repeats multiple times for multiple threads ...]
```
</details>

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

<details>

<summary>
ValueError: Sign not allowed in string format specifier
</summary>
```
Exception in thread Thread-1 (_timer_func):
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/threading.py", line 1041, in _bootstrap_inner
    self.run()
    ~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/threading.py", line 992, in run
    self._target(*self._args, **self._kwargs)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/progress.py", line 128, in _timer_func
    self._update_bar(elapsed)
    ~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/progress.py", line 134, in _update_bar
    self._draw_bar(0, elapsed)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/progress.py", line 147, in _draw_bar
    msg = "\r[{0:<{1}}] | {2}% Completed | {3}".format(
        bar, self._width, percent, elapsed
    )
ValueError: Sign not allowed in string format specifier
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/repo.py", line 12, in <module>
    result = get(dsk, 'y')
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/threaded.py", line 91, in get
    results = get_async(
        pool.submit,
    ...<6 lines>...
        **kwargs,
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/local.py", line 561, in get_async
    finish(dsk, state, not succeeded)
    ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/progress.py", line 116, in _finish
    self._draw_bar(1, elapsed)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/progress.py", line 147, in _draw_bar
    msg = "\r[{0:<{1}}] | {2}% Completed | {3}".format(
        bar, self._width, percent, elapsed
    )
ValueError: Sign not allowed in string format specifier
```
</details>

## Why This Is A Bug

This violates the principle of fail-fast error handling. The ProgressBar constructor accepts negative width values without any validation, but the error only surfaces later when the progress bar attempts to render itself in the `_draw_bar` method at line 147.

The format string `"{0:<{1}}"` uses Python's string formatting mini-language where `{1}` represents the field width for left-aligned text. Python's format specification requires that field widths must be non-negative integers. When a negative value is provided, Python raises `ValueError: Sign not allowed in string format specifier`.

The documentation states that `width` is an "int, optional - Width of the bar" with a default of 40, but does not specify that it must be positive. While semantically a negative width doesn't make sense for a progress bar, the code should either:
1. Validate the input at construction time and raise a clear error
2. Document the constraint that width must be positive

Currently, the error occurs in both the background timer thread and the main thread, causing multiple confusing error messages that don't clearly indicate the root cause (negative width parameter).

## Relevant Context

- The bug affects dask/diagnostics/progress.py in the ProgressBar class
- The issue occurs in Python's string formatting, which has been consistent across Python versions
- Width=0 works without errors (creates an empty progress bar)
- The error happens in two places: the background timer thread (_timer_func) and when finishing (_finish)
- Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/dask/diagnostics/progress.py:147`
- Documentation: https://docs.dask.org/en/stable/diagnostics-local.html

## Proposed Fix

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
+        if width < 0:
+            raise ValueError(f"width must be non-negative, got {width}")
         self._minimum = minimum
         self._width = width
```