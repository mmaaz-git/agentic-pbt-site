# Bug Report: anyio.sleep_until() Hangs Indefinitely with NaN Deadline

**Target**: `anyio.sleep_until`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`anyio.sleep_until(float('nan'))` hangs indefinitely instead of either raising an error or returning immediately. This occurs because `max(nan, 0)` returns `nan`, which is then passed to `sleep()`, causing unexpected behavior.

## Property-Based Test

```python
import math
import anyio
from hypothesis import given, strategies as st

@given(st.floats(allow_nan=True, allow_infinity=True))
def test_sleep_until_should_not_hang(deadline):
    async def run_test():
        with anyio.fail_after(1.0):
            await anyio.sleep_until(deadline)

    anyio.run(run_test)
```

**Failing input**: `deadline=nan`

## Reproducing the Bug

```python
import math
import anyio

async def test():
    print("Calling sleep_until(nan) with 1-second timeout...")
    try:
        with anyio.fail_after(1.0):
            await anyio.sleep_until(float('nan'))
        print("Completed without timeout (unexpected!)")
    except TimeoutError:
        print("Timed out - sleep_until(nan) hung indefinitely")

anyio.run(test)
```

## Why This Is A Bug

The implementation at `anyio/_core/_eventloop.py:113`:

```python
async def sleep_until(deadline: float) -> None:
    now = current_time()
    await sleep(max(deadline - now, 0))
```

When `deadline=nan`:
- `deadline - now = nan`
- `max(nan, 0)` returns `nan` in Python
- `sleep(nan)` exhibits undefined behavior (hangs or sleeps for a very long time)

Expected behavior: `sleep_until` should either:
1. Validate the deadline and raise `ValueError` for NaN/invalid values
2. Treat NaN as an immediate wake-up (delay of 0)

Actual behavior: Hangs indefinitely

## Fix

```diff
--- a/anyio/_core/_eventloop.py
+++ b/anyio/_core/_eventloop.py
@@ -110,6 +110,8 @@ async def sleep_until(deadline: float) -> None:

     """
     now = current_time()
-    await sleep(max(deadline - now, 0))
+    delay = deadline - now
+    if math.isnan(delay):
+        delay = 0
+    await sleep(max(delay, 0))
```

Alternatively, validate the input:

```diff
--- a/anyio/_core/_eventloop.py
+++ b/anyio/_core/_eventloop.py
@@ -110,5 +110,7 @@ async def sleep_until(deadline: float) -> None:

     """
+    if math.isnan(deadline):
+        raise ValueError("deadline must not be NaN")
     now = current_time()
     await sleep(max(deadline - now, 0))
```