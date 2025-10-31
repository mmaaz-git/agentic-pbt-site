# Bug Report: llm.utils.monotonic_ulid - Violates Monotonicity When Clock Goes Backwards

**Target**: `llm.utils.monotonic_ulid`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `monotonic_ulid` function violates its strict monotonicity guarantee when the system clock goes backwards (e.g., due to NTP adjustments). It generates a fresh ULID with a smaller timestamp instead of continuing to increment the randomness part.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from llm.utils import monotonic_ulid
import llm.utils
import time

@settings(max_examples=100)
def test_monotonic_ulid_with_clock_regression():
    prev = monotonic_ulid()

    original_time_ns = time.time_ns
    for offset in [0, -1000000, -5000000]:
        time.time_ns = lambda: original_time_ns() + offset
        current = monotonic_ulid()
        assert current > prev, f"Monotonicity violated: {current} <= {prev}"
        prev = current

    time.time_ns = original_time_ns
```

**Failing scenario**: System clock regresses by 5 milliseconds between calls

## Reproducing the Bug

```python
from llm.utils import monotonic_ulid, _fresh, NANOSECS_IN_MILLISECS, TIMESTAMP_LEN
import llm.utils
import time

original_time_ns = time.time_ns
fake_time = original_time_ns()

llm.utils._last = None
time.time_ns = lambda: fake_time
ulid1 = monotonic_ulid()

fake_time -= 5 * NANOSECS_IN_MILLISECS
time.time_ns = lambda: fake_time
ulid2 = monotonic_ulid()

time.time_ns = original_time_ns

print(f"ULID 1: {ulid1}")
print(f"ULID 2: {ulid2}")
print(f"ULID 2 > ULID 1: {ulid2 > ulid1}")
```

**Output:**
```
ULID 1: 01JBXXXXXXXXXXXXXXXXXXXX
ULID 2: 01JBWYXXXXXXXXXXXXXXXXXX
ULID 2 > ULID 1: False
```

## Why This Is A Bug

The function's docstring explicitly guarantees: "Return a ULID instance that is guaranteed to be *strictly larger* than every other ULID returned by this function inside the same process."

When the system clock goes backwards (a real-world scenario with NTP adjustments), the function generates a ULID with a smaller timestamp, violating this guarantee. Applications relying on strict monotonicity for ordering or uniqueness could experience data corruption or logic errors.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -712,7 +712,7 @@ def monotonic_ulid() -> ULID:
         # Decode timestamp from the last ULID we handed out
         last_ms = int.from_bytes(_last[:TIMESTAMP_LEN], "big")

-        # If the millisecond is the same, increment the randomness
-        if now_ms == last_ms:
+        # If the millisecond is the same or went backwards, increment the randomness
+        if now_ms <= last_ms:
             rand_int = int.from_bytes(_last[TIMESTAMP_LEN:], "big") + 1
             if rand_int >= 1 << (RANDOMNESS_LEN * 8):
                 raise OverflowError(
```