# Bug Report: llm.utils.monotonic_ulid Monotonicity Violation on Clock Regression

**Target**: `llm.utils.monotonic_ulid`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `monotonic_ulid` function violates its monotonicity guarantee when the system clock moves backward (e.g., due to NTP adjustment or manual clock changes). The function promises to return ULIDs that are "strictly larger" than all previously returned ULIDs, but it generates ULIDs based on the current timestamp without checking if the clock has regressed.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import monotonic_ulid
import time

def test_monotonic_ulid_always_increasing():
    ulids = []
    for _ in range(1000):
        ulid = monotonic_ulid()
        if ulids:
            assert ulid > ulids[-1], "ULID not strictly increasing"
        ulids.append(ulid)
```

**Failing scenario**: System clock moves backward between calls to `monotonic_ulid()`.

## Reproducing the Bug

```python
from llm.utils import monotonic_ulid, _fresh, _last, _lock, TIMESTAMP_LEN
import time

ulid1 = monotonic_ulid()
print(f"First ULID: {ulid1}")

with _lock:
    last_ms = int.from_bytes(_last[:TIMESTAMP_LEN], "big")
    print(f"Timestamp in ULID: {last_ms}")

    backward_ms = last_ms - 1000
    fake_ulid = _fresh(backward_ms)

    new_ulid_bytes = fake_ulid
    from ulid import ULID
    ulid2 = ULID(new_ulid_bytes)

print(f"ULID after clock regression: {ulid2}")
print(f"Monotonicity violated: {ulid2 < ulid1}")
```

## Why This Is A Bug

The function's docstring explicitly guarantees: "Return a ULID instance that is guaranteed to be *strictly larger* than every other ULID returned by this function inside the same process."

However, when the system clock moves backward:
- `now_ms < last_ms` (current time is before the last ULID's timestamp)
- The code generates a fresh ULID with `now_ms` as the timestamp
- This new ULID has a smaller timestamp than the previous ULID
- Since ULIDs are compared lexicographically with timestamp as the most significant bits, the new ULID is smaller
- This violates the monotonicity guarantee

System clocks can move backward due to:
1. NTP (Network Time Protocol) synchronization
2. Manual clock adjustments
3. Virtual machine clock corrections
4. Hardware clock drift corrections

## Fix

The function should detect clock regression and handle it by either:
1. Using the last timestamp and incrementing the randomness (treating it as if we're still in the same millisecond)
2. Raising an error to alert the caller
3. Waiting until the clock catches up

Here's a fix using approach 1:

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -713,6 +713,11 @@ def monotonic_ulid() -> ULID:
         last_ms = int.from_bytes(_last[:TIMESTAMP_LEN], "big")

         # If the millisecond is the same, increment the randomness
+        # Also handle clock regression by treating it as same millisecond
+        if now_ms < last_ms:
+            # Clock moved backward, use last_ms to maintain monotonicity
+            now_ms = last_ms
+
         if now_ms == last_ms:
             rand_int = int.from_bytes(_last[TIMESTAMP_LEN:], "big") + 1
             if rand_int >= 1 << (RANDOMNESS_LEN * 8):
```