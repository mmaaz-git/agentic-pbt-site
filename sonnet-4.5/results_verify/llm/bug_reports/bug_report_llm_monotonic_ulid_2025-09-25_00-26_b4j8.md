# Bug Report: llm.utils.monotonic_ulid Violates Monotonicity with Clock Skew

**Target**: `llm.utils.monotonic_ulid`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `monotonic_ulid` function violates its monotonicity guarantee when the system clock goes backwards (e.g., due to NTP adjustments or manual time changes). It generates ULIDs with earlier timestamps than previously generated ULIDs, breaking the "strictly larger" invariant.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import monotonic_ulid
import time


def test_monotonic_ulid_strictly_increasing_with_time_manipulation():
    ulids = []

    ulid1 = monotonic_ulid()
    ulids.append(ulid1)

    for _ in range(10):
        ulid_next = monotonic_ulid()
        ulids.append(ulid_next)

        assert ulid_next > ulids[-2], f"Monotonicity violated: {ulid_next} <= {ulids[-2]}"
```

**Note**: This test would fail if the system clock goes backwards during execution, which can happen in production systems.

## Reproducing the Bug

The bug cannot be easily reproduced without manipulating system time, but the logic flaw is evident in the code:

```python
def monotonic_ulid() -> ULID:
    global _last
    now_ms = time.time_ns() // NANOSECS_IN_MILLISECS

    with _lock:
        if _last is None:
            _last = _fresh(now_ms)
            return ULID(_last)

        last_ms = int.from_bytes(_last[:TIMESTAMP_LEN], "big")

        if now_ms == last_ms:
            # Increment randomness
            ...
        # BUG: Missing check for now_ms < last_ms!
        # New millisecond, start fresh
        _last = _fresh(now_ms)  # This can have earlier timestamp!
        return ULID(_last)
```

**Scenario:**
1. Call `monotonic_ulid()` at time T=1000ms → generates ULID with timestamp 1000
2. System clock goes backwards to T=999ms (NTP adjustment)
3. Call `monotonic_ulid()` at time T=999ms → generates ULID with timestamp 999
4. ULID from step 3 < ULID from step 1 (monotonicity violated!)

## Why This Is A Bug

The function's docstring explicitly promises:

> "Return a ULID instance that is guaranteed to be *strictly larger* than every other ULID returned by this function inside the same process."

This guarantee is violated when the system clock goes backwards. This is a real-world scenario that can happen due to:
- NTP synchronization adjusting the clock
- Manual time changes by administrators
- Virtualization/containerization time skew
- Leap second adjustments

## Fix

```diff
    # Decode timestamp from the last ULID we handed out
    last_ms = int.from_bytes(_last[:TIMESTAMP_LEN], "big")

    # If the millisecond is the same, increment the randomness
    if now_ms == last_ms:
        rand_int = int.from_bytes(_last[TIMESTAMP_LEN:], "big") + 1
        if rand_int >= 1 << (RANDOMNESS_LEN * 8):
            raise OverflowError(
                "Randomness overflow: > 2**80 ULIDs requested "
                "in one millisecond!"
            )
        randomness = rand_int.to_bytes(RANDOMNESS_LEN, "big")
        _last = _last[:TIMESTAMP_LEN] + randomness
        return ULID(_last)
+
+   # If clock went backwards, continue from last timestamp
+   if now_ms < last_ms:
+       # Treat it as if we're in the same millisecond
+       rand_int = int.from_bytes(_last[TIMESTAMP_LEN:], "big") + 1
+       if rand_int >= 1 << (RANDOMNESS_LEN * 8):
+           raise OverflowError(
+               "Clock went backwards and randomness overflow"
+           )
+       randomness = rand_int.to_bytes(RANDOMNESS_LEN, "big")
+       _last = _last[:TIMESTAMP_LEN] + randomness
+       return ULID(_last)

    # New millisecond, start fresh
    _last = _fresh(now_ms)
    return ULID(_last)
```