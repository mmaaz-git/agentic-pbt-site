# Bug Report: llm.utils.monotonic_ulid Breaks Strict Monotonicity When Clock Goes Backwards

**Target**: `llm.utils.monotonic_ulid`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `monotonic_ulid` function violates its documented guarantee of generating "strictly larger" ULIDs when the system clock moves backwards, producing ULIDs with earlier timestamps that compare as less than previously generated ULIDs.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to discover the monotonic_ulid bug.
This test simulates clock skew scenarios to verify monotonicity guarantees.
"""

import sys
import time
from unittest import mock

# Add the llm package to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
from llm.utils import monotonic_ulid


@given(time_sequence=st.lists(
    st.integers(min_value=1, max_value=10000),  # milliseconds
    min_size=2,
    max_size=10
))
@example(time_sequence=[1000, 1001, 999])  # The specific failing case
@settings(deadline=1000)
def test_monotonic_ulid_strictly_increasing_with_time_manipulation(time_sequence):
    """
    Property-based test that verifies monotonic_ulid maintains strict monotonicity
    even when the system clock moves backwards (clock skew).

    The function's documentation guarantees that each ULID will be "strictly larger"
    than every other ULID returned by this function inside the same process.
    """
    ulids = []

    # Mock time.time_ns to control the clock
    time_index = [0]

    def controlled_time_ns():
        # Convert milliseconds to nanoseconds
        return time_sequence[time_index[0]] * 1000000

    with mock.patch('time.time_ns', side_effect=controlled_time_ns):
        for i in range(len(time_sequence)):
            time_index[0] = i
            ulid = monotonic_ulid()
            ulids.append(ulid)

            # Verify strict monotonicity: each ULID must be greater than all previous
            for j, prev_ulid in enumerate(ulids[:-1]):
                assert ulid > prev_ulid, (
                    f"Monotonicity violated at index {i}: "
                    f"ULID #{i+1} ({ulid}) is not greater than "
                    f"ULID #{j+1} ({prev_ulid}). "
                    f"Time sequence: {time_sequence[:i+1]}ms"
                )


if __name__ == "__main__":
    # Run the test
    print("Running property-based test for monotonic_ulid...")
    print("=" * 60)

    try:
        test_monotonic_ulid_strictly_increasing_with_time_manipulation()
        print("✅ All tests passed!")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        print("\nThis demonstrates that monotonic_ulid violates its documented")
        print("guarantee of strict monotonicity when the clock goes backwards.")
```

<details>

<summary>
**Failing input**: `time_sequence=[1000, 1001, 999]`
</summary>
```
Running property-based test for monotonic_ulid...
============================================================
❌ Test failed: Monotonicity violated at index 2: ULID #3 (00000000Z7JRTRAXXCAVW0ANPM) is not greater than ULID #1 (00000000Z8FY753R1PKX7KY2X2). Time sequence: [1000, 1001, 999]ms

This demonstrates that monotonic_ulid violates its documented
guarantee of strict monotonicity when the clock goes backwards.
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Demonstration of the monotonic_ulid bug when system clock goes backwards.
This simulates the scenario where NTP adjustments or other clock changes
cause time to move backwards.
"""

import sys
import os
import time
from unittest import mock

# Add the llm package to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import monotonic_ulid

def test_monotonic_ulid_with_clock_skew():
    """
    Test that demonstrates the monotonic_ulid bug when clock goes backwards.
    """
    print("Testing monotonic_ulid behavior with clock skew...")
    print("-" * 60)

    # Store generated ULIDs
    ulids = []

    # Mock time to control clock behavior
    original_time_ns = time.time_ns

    # Start at time 1000ms (1000000000 nanoseconds)
    mock_time = 1000 * 1000000  # 1000ms in nanoseconds

    def controlled_time_ns():
        return mock_time

    with mock.patch('time.time_ns', side_effect=controlled_time_ns):
        # Generate first ULID at T=1000ms
        print(f"Time T=1000ms: Generating ULID #1")
        ulid1 = monotonic_ulid()
        ulids.append(ulid1)
        print(f"  ULID #1: {ulid1}")
        print(f"  ULID #1 bytes: {ulid1.bytes.hex()}")

        # Move time forward to 1001ms
        mock_time = 1001 * 1000000
        print(f"\nTime T=1001ms: Generating ULID #2")
        ulid2 = monotonic_ulid()
        ulids.append(ulid2)
        print(f"  ULID #2: {ulid2}")
        print(f"  ULID #2 bytes: {ulid2.bytes.hex()}")

        # Verify monotonicity so far
        assert ulid2 > ulid1, f"ULID #2 should be > ULID #1"
        print(f"  ✓ ULID #2 > ULID #1: {ulid2 > ulid1}")

        # SIMULATE CLOCK GOING BACKWARDS (e.g., NTP adjustment)
        mock_time = 999 * 1000000  # Go back to 999ms
        print(f"\n⚠️  CLOCK SKEW: Time goes backwards to T=999ms")
        print(f"Time T=999ms: Generating ULID #3")
        ulid3 = monotonic_ulid()
        ulids.append(ulid3)
        print(f"  ULID #3: {ulid3}")
        print(f"  ULID #3 bytes: {ulid3.bytes.hex()}")

        # Check if monotonicity is maintained
        print(f"\n🔍 Checking monotonicity...")
        print(f"  ULID #3 > ULID #2: {ulid3 > ulid2}")
        print(f"  ULID #3 > ULID #1: {ulid3 > ulid1}")

        if ulid3 <= ulid2:
            print(f"\n❌ BUG DETECTED: Monotonicity violated!")
            print(f"   ULID #3 ({ulid3}) is not greater than ULID #2 ({ulid2})")
            print(f"   This violates the documented guarantee that each ULID")
            print(f"   must be 'strictly larger' than all previous ULIDs.")

            # Show the timestamp portions
            ts1 = int.from_bytes(ulid1.bytes[:6], "big")
            ts2 = int.from_bytes(ulid2.bytes[:6], "big")
            ts3 = int.from_bytes(ulid3.bytes[:6], "big")
            print(f"\n📊 Timestamp Analysis:")
            print(f"   ULID #1 timestamp: {ts1}ms")
            print(f"   ULID #2 timestamp: {ts2}ms")
            print(f"   ULID #3 timestamp: {ts3}ms")
            print(f"   Note: ULID #3 has an earlier timestamp than ULID #2!")
        else:
            print(f"\n✅ Monotonicity maintained (no bug)")

if __name__ == "__main__":
    try:
        test_monotonic_ulid_with_clock_skew()
    except Exception as e:
        print(f"\n💥 Exception occurred: {e}")
        import traceback
        traceback.print_exc()
```

<details>

<summary>
Monotonicity violation when clock goes backwards from 1001ms to 999ms
</summary>
```
Testing monotonic_ulid behavior with clock skew...
------------------------------------------------------------
Time T=1000ms: Generating ULID #1
  ULID #1: 00000000Z8KZB5T6XS3B2XCXP1
  ULID #1 bytes: 0000000003e89fd65d1bb91ac5d676c1

Time T=1001ms: Generating ULID #2
  ULID #2: 00000000Z9N1Q4JVSH12SJ6CE5
  ULID #2 bytes: 0000000003e9a86e496f3108b32331c5
  ✓ ULID #2 > ULID #1: True

⚠️  CLOCK SKEW: Time goes backwards to T=999ms
Time T=999ms: Generating ULID #3
  ULID #3: 00000000Z715TR3YS1SZ14KWRZ
  ULID #3 bytes: 0000000003e7097581fb21cfc249f31f

🔍 Checking monotonicity...
  ULID #3 > ULID #2: False
  ULID #3 > ULID #1: False

❌ BUG DETECTED: Monotonicity violated!
   ULID #3 (00000000Z715TR3YS1SZ14KWRZ) is not greater than ULID #2 (00000000Z9N1Q4JVSH12SJ6CE5)
   This violates the documented guarantee that each ULID
   must be 'strictly larger' than all previous ULIDs.

📊 Timestamp Analysis:
   ULID #1 timestamp: 1000ms
   ULID #2 timestamp: 1001ms
   ULID #3 timestamp: 999ms
   Note: ULID #3 has an earlier timestamp than ULID #2!
```
</details>

## Why This Is A Bug

The `monotonic_ulid` function's docstring explicitly guarantees that it will "Return a ULID instance that is guaranteed to be *strictly larger* than every other ULID returned by this function inside the same process." This is an unqualified, absolute guarantee with no mentioned exceptions.

The bug occurs because the implementation has only three code paths:
1. First call (`_last is None`): Generates a fresh ULID
2. Same millisecond (`now_ms == last_ms`): Increments the randomness portion to maintain order
3. Different millisecond: Generates a fresh ULID with the current timestamp

The critical flaw is in path 3: there is no check for `now_ms < last_ms`. When the system clock moves backwards, the code generates a new ULID with the earlier timestamp. Since ULIDs are compared lexicographically with the timestamp as the most significant portion (first 48 bits), a ULID with an earlier timestamp will always be smaller than one with a later timestamp, regardless of the randomness portion.

This violates the strict monotonicity guarantee in real-world scenarios including:
- NTP (Network Time Protocol) synchronization adjustments
- Virtual machine migrations or suspensions
- Manual system time changes by administrators
- Leap second adjustments
- Container/virtualization time skew

The documentation states the function works "the same way the reference JavaScript `monotonicFactory` does" but only describes behavior for same-millisecond and forward-moving time, leaving backward time movement unspecified and incorrectly handled.

## Relevant Context

The bug is in `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py` at lines 726-729. The function correctly handles incrementing randomness when multiple ULIDs are generated in the same millisecond (lines 715-725) but fails to handle the case where `now_ms < last_ms`.

The ULID specification (https://github.com/ulid/spec) consists of:
- 48-bit timestamp (millisecond precision Unix time)
- 80-bit randomness

ULIDs are lexicographically sortable because the timestamp is the most significant portion. This is why a ULID with timestamp 999ms will always be less than one with timestamp 1001ms, breaking monotonicity when clocks go backward.

The JavaScript reference implementation mentioned in the docstring (https://github.com/ulid/javascript) does handle clock regression by treating it similarly to the same-millisecond case, incrementing the randomness while keeping the higher timestamp.

## Proposed Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -722,6 +722,18 @@ def monotonic_ulid() -> ULID:
             randomness = rand_int.to_bytes(RANDOMNESS_LEN, "big")
             _last = _last[:TIMESTAMP_LEN] + randomness
             return ULID(_last)
+
+        # If clock went backwards, maintain monotonicity by keeping
+        # the previous timestamp and incrementing randomness
+        if now_ms < last_ms:
+            rand_int = int.from_bytes(_last[TIMESTAMP_LEN:], "big") + 1
+            if rand_int >= 1 << (RANDOMNESS_LEN * 8):
+                raise OverflowError(
+                    "Randomness overflow: clock went backwards and "
+                    "> 2**80 ULIDs requested for previous timestamp"
+                )
+            randomness = rand_int.to_bytes(RANDOMNESS_LEN, "big")
+            _last = _last[:TIMESTAMP_LEN] + randomness
+            return ULID(_last)

         # New millisecond, start fresh
         _last = _fresh(now_ms)
```