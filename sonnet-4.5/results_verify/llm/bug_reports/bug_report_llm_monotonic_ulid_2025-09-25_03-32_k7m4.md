# Bug Report: llm.utils.monotonic_ulid Clock Backward Violation

**Target**: `llm.utils.monotonic_ulid`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `monotonic_ulid()` function violates its monotonicity guarantee when the system clock moves backward, returning a ULID that is smaller than a previously returned one.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from unittest import mock
import time
from llm.utils import monotonic_ulid, NANOSECS_IN_MILLISECS

@given(st.integers(min_value=1, max_value=1000))
def test_monotonic_ulid_clock_backward(backward_ms):
    ulid1 = monotonic_ulid()

    current_time = time.time_ns()
    backward_time = current_time - (backward_ms * NANOSECS_IN_MILLISECS)

    with mock.patch('time.time_ns', return_value=backward_time):
        ulid2 = monotonic_ulid()

    assert ulid1 < ulid2, f"Monotonicity violated: {ulid1} >= {ulid2}"
```

**Failing input**: Any scenario where `time.time_ns()` returns a value smaller than the previous call (e.g., `backward_ms=1`).

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import time
from unittest import mock
from llm.utils import monotonic_ulid, NANOSECS_IN_MILLISECS

ulid1 = monotonic_ulid()
print(f"First ULID: {ulid1}")

current_time_ns = time.time_ns()
backward_time_ns = current_time_ns - (2 * NANOSECS_IN_MILLISECS)

with mock.patch('time.time_ns', return_value=backward_time_ns):
    ulid2 = monotonic_ulid()
    print(f"Second ULID (after clock went backward): {ulid2}")

    print(f"\nulid1 < ulid2: {ulid1 < ulid2}")
    print(f"ulid1 >= ulid2: {ulid1 >= ulid2}")

    if ulid1 >= ulid2:
        print("BUG: Monotonicity violated!")
```

## Why This Is A Bug

The function's docstring explicitly states (lines 691-692):

> Return a ULID instance that is guaranteed to be *strictly larger* than every other ULID returned by this function inside the same process.

However, when the system clock moves backward (a realistic scenario due to NTP adjustments, clock corrections, VM synchronization, etc.), the function generates a new ULID with a timestamp smaller than the previous one. Since ULIDs are compared primarily by their timestamp component (first 48 bits), this violates the strict monotonicity guarantee.

The code handles two cases:
1. `now_ms == last_ms`: Increments randomness ✓
2. `now_ms != last_ms`: Generates fresh ULID with `now_ms`

But case #2 includes both `now_ms > last_ms` (correct) and `now_ms < last_ms` (incorrect - violates monotonicity).

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -713,7 +713,7 @@ def monotonic_ulid() -> ULID:
         last_ms = int.from_bytes(_last[:TIMESTAMP_LEN], "big")

-        # If the millisecond is the same, increment the randomness
-        if now_ms == last_ms:
+        # If the millisecond is the same or clock went backward, increment the randomness
+        if now_ms <= last_ms:
             rand_int = int.from_bytes(_last[TIMESTAMP_LEN:], "big") + 1
             if rand_int >= 1 << (RANDOMNESS_LEN * 8):
```

This fix ensures that when the clock moves backward (`now_ms < last_ms`), the function uses the previous timestamp and increments the randomness portion, maintaining strict monotonicity.