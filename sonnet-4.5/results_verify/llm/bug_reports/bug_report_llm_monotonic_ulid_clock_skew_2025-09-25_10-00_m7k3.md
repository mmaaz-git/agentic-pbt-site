# Bug Report: llm.utils.monotonic_ulid Clock Skew Violation

**Target**: `llm.utils.monotonic_ulid`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `monotonic_ulid` function violates its documented guarantee of returning "strictly larger" ULIDs when the system clock moves backwards (due to NTP adjustments, manual changes, or other clock skew).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from unittest.mock import patch
import llm.utils

@given(st.lists(st.integers(min_value=0, max_value=2**48-1), min_size=2, max_size=100))
def test_monotonic_ulid_always_increasing(timestamps_ms):
    """ULIDs should be strictly increasing even if timestamps aren't monotonic"""
    llm.utils._last = None
    ulids = []

    for ts in timestamps_ms:
        with patch('time.time_ns', return_value=ts * 1_000_000):
            ulid = llm.utils.monotonic_ulid()
            ulids.append(bytes(ulid))

    for i in range(len(ulids) - 1):
        assert ulids[i] < ulids[i+1], f"ULID {i+1} not strictly greater than ULID {i}"
```

**Failing input**: `timestamps_ms = [1000, 999]` (clock goes backwards)

## Reproducing the Bug

```python
import llm.utils
from unittest.mock import patch

llm.utils._last = None

with patch('time.time_ns', return_value=1000_000_000):
    ulid1 = llm.utils.monotonic_ulid()
    print(f"ULID 1: {ulid1}")

with patch('time.time_ns', return_value=999_000_000):
    ulid2 = llm.utils.monotonic_ulid()
    print(f"ULID 2: {ulid2}")

if bytes(ulid2) > bytes(ulid1):
    print("✅ Monotonicity preserved")
else:
    print("❌ ULID 2 <= ULID 1 (monotonicity violated!)")
```

**Expected**: ULID 2 > ULID 1 (as promised by docstring)
**Actual**: ULID 2 < ULID 1 (when clock goes backwards)

## Why This Is A Bug

The function's docstring explicitly promises:

> "Return a ULID instance that is guaranteed to be *strictly larger* than every other ULID returned by this function inside the same process."

The implementation uses `time.time_ns()` which returns wall-clock time that can move backwards due to:
- NTP clock synchronization
- Manual system clock adjustments
- Daylight saving time changes
- Virtualization/container time synchronization

When the clock moves backwards, the code generates a new ULID with an earlier timestamp (line 728):
```python
_last = _fresh(now_ms)  # now_ms < last_ms, violating monotonicity
```

This creates a ULID that is lexicographically smaller than the previous one, directly contradicting the documented guarantee.

## Fix

Use `time.monotonic_ns()` instead of `time.time_ns()` to ensure the time source never decreases:

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -701,7 +701,7 @@ def monotonic_ulid() -> ULID:
     """
     global _last

-    now_ms = time.time_ns() // NANOSECS_IN_MILLISECS
+    now_ms = time.monotonic_ns() // NANOSECS_IN_MILLISECS

     with _lock:
         # First call
```

**Note**: This changes ULIDs to use monotonic time instead of wall-clock time. If wall-clock time is required for interoperability, an alternative fix is to detect and handle backwards clock movement:

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -713,6 +713,11 @@ def monotonic_ulid() -> ULID:
         # Decode timestamp from the last ULID we handed out
         last_ms = int.from_bytes(_last[:TIMESTAMP_LEN], "big")

+        # Handle backwards clock movement
+        if now_ms < last_ms:
+            now_ms = last_ms
+
         # If the millisecond is the same, increment the randomness
         if now_ms == last_ms:
```