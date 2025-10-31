# Bug Report: anyio fail_after/move_on_after NaN Crash

**Target**: `anyio._core._tasks.fail_after` and `anyio._core._tasks.move_on_after`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `fail_after()` and `move_on_after()` functions crash with a confusing `ValueError: cannot convert float NaN to integer` when passed `math.nan` as the delay parameter, instead of validating the input or handling it gracefully.

## Property-Based Test

```python
import math

import pytest
from hypothesis import given, settings, strategies as st


@given(st.floats(allow_nan=True))
@settings(max_examples=200)
@pytest.mark.asyncio
async def test_fail_after_accepts_valid_numeric_delays(delay):
    import anyio
    from anyio import fail_after

    if math.isnan(delay):
        with pytest.raises((ValueError, TypeError)):
            with fail_after(delay):
                await anyio.sleep(0)
    else:
        with fail_after(delay):
            await anyio.sleep(0)
```

**Failing input**: `math.nan`

## Reproducing the Bug

```python
import asyncio
import math

import anyio
from anyio import fail_after


async def main():
    with fail_after(math.nan):
        await anyio.sleep(0.01)


asyncio.run(main())
```

## Why This Is A Bug

When `fail_after(math.nan)` is called:
1. The function computes `deadline = current_time() + math.nan`, which produces `nan`
2. This NaN deadline gets passed to the cancel scope
3. The event loop's selector eventually tries to convert the NaN timeout to an integer
4. This causes `ValueError: cannot convert float NaN to integer` deep in the selector code

The error message is confusing and doesn't indicate that the root cause is an invalid delay parameter. Users would expect either:
- Input validation that rejects NaN with a clear error message
- Graceful handling of NaN (e.g., treating it as no timeout)

The same bug affects `move_on_after()`.

## Fix

Add input validation to reject NaN values with a clear error message:

```diff
--- a/anyio/_core/_tasks.py
+++ b/anyio/_core/_tasks.py
@@ -108,6 +108,8 @@ def fail_after(

     """
     current_time = get_async_backend().current_time
+    if delay is not None and math.isnan(delay):
+        raise ValueError("delay must not be NaN")
     deadline = (current_time() + delay) if delay is not None else math.inf
     with get_async_backend().create_cancel_scope(
         deadline=deadline, shield=shield
@@ -129,6 +131,8 @@ def move_on_after(delay: float | None, shield: bool = False) -> CancelScope:
     :return: a cancel scope

     """
+    if delay is not None and math.isnan(delay):
+        raise ValueError("delay must not be NaN")
     deadline = (
         (get_async_backend().current_time() + delay) if delay is not None else math.inf
     )
```