# Bug Report: anyio fail_after/move_on_after NaN Input Validation Failure

**Target**: `anyio._core._tasks.fail_after` and `anyio._core._tasks.move_on_after`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `fail_after()` and `move_on_after()` functions crash with a confusing `ValueError: cannot convert float NaN to integer` from deep in the asyncio selector code when passed `math.nan` as the delay parameter, instead of validating the input at the API boundary.

## Property-Based Test

```python
import math

import pytest
from hypothesis import given, settings, strategies as st


@given(st.one_of(
    st.just(math.nan),
    st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False)
))
@settings(max_examples=200)
@pytest.mark.asyncio
async def test_fail_after_handles_all_float_values(delay):
    import anyio
    from anyio import fail_after

    # This test demonstrates the bug: NaN causes an unexpected crash deep in the event loop
    if math.isnan(delay):
        # The bug: This causes "ValueError: cannot convert float NaN to integer"
        # from deep in the selector code, not at the API boundary
        try:
            with fail_after(delay):
                await anyio.sleep(0.01)
        except ValueError as e:
            # This is the bug - we get a confusing error from deep in the stack
            assert "cannot convert float NaN to integer" in str(e)
            # A better behavior would be to validate at the API boundary
    else:
        # Normal delays should work
        try:
            with fail_after(delay):
                if delay > 0.01:
                    await anyio.sleep(0.001)
                else:
                    pass
        except TimeoutError:
            pass  # Expected for small delays
```

<details>

<summary>
**Failing input**: `delay=nan`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/3
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_fail_after_handles_all_float_values FAILED                 [100%]

=================================== FAILURES ===================================
___________________ test_fail_after_handles_all_float_values ___________________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 8, in test_fail_after_handles_all_float_values
  |     st.just(math.nan),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | hypothesis.errors.FlakyFailure: Hypothesis test_fail_after_handles_all_float_values(delay=nan) produces unreliable results: Falsified on the first call but did not on a subsequent one (1 sub-exception)
  | Falsifying example: test_fail_after_handles_all_float_values(
  |     delay=nan,
  | )
  | Failed to reproduce exception. Expected:
  | /home/npc/miniconda/lib/python3.13/site-packages/pytest_asyncio/plugin.py:721: in inner
  |     runner.run(coro, context=context)
  | /home/npc/miniconda/lib/python3.13/asyncio/runners.py:118: in run
  |     return self._loop.run_until_complete(task)
  |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  | /home/npc/miniconda/lib/python3.13/asyncio/base_events.py:712: in run_until_complete
  |     self.run_forever()
  | /home/npc/miniconda/lib/python3.13/asyncio/base_events.py:683: in run_forever
  |     self._run_once()
  | /home/npc/miniconda/lib/python3.13/asyncio/base_events.py:2002: in _run_once
  |     event_list = self._selector.select(timeout)
  |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  | /home/npc/miniconda/lib/python3.13/selectors.py:443: in select
  |     timeout = math.ceil(timeout * 1e3) * 1e-3
  |               ^^^^^^^^^^^^^^^^^^^^^^^^
  | E   ValueError: cannot convert float NaN to integer
  |
  | Explanation:
  |     These lines were always and only run by failing examples:
  |         /home/npc/pbt/agentic-pbt/worker_/3/hypo.py:21
  |         /home/npc/pbt/agentic-pbt/worker_/3/hypo.py:23
  |         /home/npc/miniconda/lib/python3.13/asyncio/base_events.py:713
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1207, in _execute_once_for_engine
    |     result = self.execute_once(data)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1147, in execute_once
    |     result = self.test_runner(data, run)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 822, in default_executor
    |     return function(data)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1103, in run
    |     return test(*args, **kwargs)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pytest_asyncio/plugin.py", line 719, in test_fail_after_handles_all_float_values
    |     def inner(*args, **kwargs):
    |                ^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pytest_asyncio/plugin.py", line 721, in inner
    |     runner.run(coro, context=context)
    |     ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 118, in run
    |     return self._loop.run_until_complete(task)
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 712, in run_until_complete
    |     self.run_forever()
    |     ~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 683, in run_forever
    |     self._run_once()
    |     ~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 2002, in _run_once
    |     event_list = self._selector.select(timeout)
    |   File "/home/npc/miniconda/lib/python3.13/selectors.py", line 443, in select
    |     timeout = math.ceil(timeout * 1e3) * 1e-3
    |               ~~~~~~~~~^^^^^^^^^^^^^^^
    | ValueError: cannot convert float NaN to integer
    +------------------------------------
=========================== short test summary info ============================
FAILED hypo.py::test_fail_after_handles_all_float_values - ValueError('cannot...
============================== 1 failed in 1.08s ===============================
```
</details>

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

<details>

<summary>
ValueError: cannot convert float NaN to integer
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/repo.py", line 13, in <module>
    asyncio.run(main())
    ~~~~~~~~~~~^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 712, in run_until_complete
    self.run_forever()
    ~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 683, in run_forever
    self._run_once()
    ~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 2002, in _run_once
    event_list = self._selector.select(timeout)
  File "/home/npc/miniconda/lib/python3.13/selectors.py", line 443, in select
    timeout = math.ceil(timeout * 1e3) * 1e-3
              ~~~~~~~~~^^^^^^^^^^^^^^^
ValueError: cannot convert float NaN to integer
```
</details>

## Why This Is A Bug

When `fail_after(math.nan)` is called, the function computes `deadline = current_time() + math.nan` on line 111 of `/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages/anyio/_core/_tasks.py`, which produces `nan` as the deadline. This NaN deadline propagates through the cancel scope into the asyncio event loop. Eventually, when the selector tries to wait with this timeout, it attempts to convert NaN to an integer for the timeout calculation, causing the crash at `/home/npc/miniconda/lib/python3.13/selectors.py:443`.

The bug violates expected behavior because:
1. **Poor error messaging**: The error occurs deep in the event loop with a message that doesn't indicate the actual problem (invalid delay parameter)
2. **No input validation**: The functions accept `float | None` but don't validate that the float is a finite, valid number
3. **API contract violation**: While the documentation doesn't explicitly forbid NaN, a NaN delay is semantically meaningless - you cannot wait for "not a number" seconds

Good API design requires validating inputs and providing clear error messages when validation fails. The current behavior makes debugging difficult as the stack trace doesn't point to the actual source of the problem.

## Relevant Context

- The same issue affects both `fail_after()` and `move_on_after()` functions in anyio
- The bug is somewhat flaky in tests, suggesting timing-dependent behavior
- The functions correctly handle other edge cases like negative delays (which expire immediately) and None (which means no timeout)
- Documentation for these functions: https://anyio.readthedocs.io/en/stable/api.html#anyio.fail_after
- Source code location: https://github.com/agronholm/anyio/blob/master/src/anyio/_core/_tasks.py

## Proposed Fix

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