# Bug Report: llm.utils.monotonic_ulid Violates Strict Monotonicity When System Clock Moves Backward

**Target**: `llm.utils.monotonic_ulid`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `monotonic_ulid()` function fails to maintain its documented guarantee of returning strictly larger ULIDs when the system clock moves backward, instead returning a smaller ULID that violates the monotonicity contract.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from unittest import mock
import time
from llm.utils import monotonic_ulid, NANOSECS_IN_MILLISECS

@given(st.integers(min_value=1, max_value=1000))
def test_monotonic_ulid_clock_backward(backward_ms):
    """Test that monotonic_ulid maintains strict monotonicity even when clock goes backward."""
    # Get first ULID
    ulid1 = monotonic_ulid()

    # Simulate clock going backward
    current_time = time.time_ns()
    backward_time = current_time - (backward_ms * NANOSECS_IN_MILLISECS)

    # Get second ULID with backward clock
    with mock.patch('time.time_ns', return_value=backward_time):
        ulid2 = monotonic_ulid()

    # Assert strict monotonicity
    assert ulid1 < ulid2, f"Monotonicity violated: {ulid1} >= {ulid2} when clock went backward by {backward_ms}ms"

if __name__ == "__main__":
    test_monotonic_ulid_clock_backward()
```

<details>

<summary>
**Failing input**: `backward_ms=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 27, in <module>
    test_monotonic_ulid_clock_backward()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 10, in test_monotonic_ulid_clock_backward
    def test_monotonic_ulid_clock_backward(backward_ms):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 24, in test_monotonic_ulid_clock_backward
    assert ulid1 < ulid2, f"Monotonicity violated: {ulid1} >= {ulid2} when clock went backward by {backward_ms}ms"
           ^^^^^^^^^^^^^
AssertionError: Monotonicity violated: 01K61B66K28YPG2A22P7W9SK08 >= 01K61B66K15NV1HTS3T6QHZFTZ when clock went backward by 1ms
Falsifying example: test_monotonic_ulid_clock_backward(
    backward_ms=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import time
from unittest import mock
from llm.utils import monotonic_ulid, NANOSECS_IN_MILLISECS

# Get first ULID
ulid1 = monotonic_ulid()
print(f"First ULID: {ulid1}")

# Simulate clock going backward by 2ms
current_time_ns = time.time_ns()
backward_time_ns = current_time_ns - (2 * NANOSECS_IN_MILLISECS)

# Get second ULID with backward clock
with mock.patch('time.time_ns', return_value=backward_time_ns):
    ulid2 = monotonic_ulid()
    print(f"Second ULID (after clock went backward by 2ms): {ulid2}")

# Check monotonicity
print(f"\nMonotonicity check:")
print(f"ulid1 < ulid2 (should be True): {ulid1 < ulid2}")
print(f"ulid1 >= ulid2 (should be False): {ulid1 >= ulid2}")

if ulid1 >= ulid2:
    print("\nBUG CONFIRMED: Monotonicity violated! The second ULID is not strictly larger than the first.")
else:
    print("\nNo bug: Monotonicity maintained.")
```

<details>

<summary>
Monotonicity violation confirmed: Second ULID is smaller than first
</summary>
```
First ULID: 01K61B5DATBZSP8N6C9FCHPCCN
Second ULID (after clock went backward by 2ms): 01K61B5DARNJ4T76R2VF914TQM

Monotonicity check:
ulid1 < ulid2 (should be True): False
ulid1 >= ulid2 (should be False): True

BUG CONFIRMED: Monotonicity violated! The second ULID is not strictly larger than the first.
```
</details>

## Why This Is A Bug

This violates the explicit guarantee in the function's docstring (lines 691-692 of `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py`):

> "Return a ULID instance that is guaranteed to be *strictly larger* than every other ULID returned by this function inside the same process."

The current implementation has a critical logic flaw in how it handles time changes:

1. When `now_ms == last_ms` (line 716): The function correctly increments the randomness portion to maintain monotonicity
2. When `now_ms != last_ms` (line 728): The function generates a fresh ULID with the current timestamp

The problem is that case #2 includes both forward time movement (`now_ms > last_ms`) AND backward time movement (`now_ms < last_ms`). When the clock moves backward, the function creates a ULID with a smaller timestamp than the previous one. Since ULIDs are lexicographically ordered with the timestamp as the most significant component (first 48 bits), this results in a smaller ULID, violating the strict monotonicity guarantee.

Clock backward movement is a realistic scenario in production systems due to:
- NTP (Network Time Protocol) adjustments
- VM migrations and synchronization
- Manual clock corrections
- Daylight saving time transitions in misconfigured systems
- Container/Docker clock drift corrections

The documentation also claims (line 694) that the function "works the same way the reference JavaScript `monotonicFactory` does", but the JavaScript reference implementation handles backward clocks correctly by maintaining monotonicity even when timestamps decrease.

## Relevant Context

- The ULID type is imported from the external `ulid` package (line 17)
- ULIDs are 128-bit identifiers with 48-bit timestamp + 80-bit randomness
- The function uses a global `_last` variable protected by a lock to track the previous ULID
- The timestamp component uses millisecond precision (via `NANOSECS_IN_MILLISECS = 1000000`)
- The function already handles overflow when more than 2^80 ULIDs are requested in the same millisecond

Key code locations:
- Function definition: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py:689-729`
- Timestamp extraction: line 713
- Same-millisecond handling: lines 716-725
- Different-millisecond handling: lines 728-729

## Proposed Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -713,8 +713,8 @@ def monotonic_ulid() -> ULID:
         # Decode timestamp from the last ULID we handed out
         last_ms = int.from_bytes(_last[:TIMESTAMP_LEN], "big")

-        # If the millisecond is the same, increment the randomness
-        if now_ms == last_ms:
+        # If the millisecond is the same or clock went backward, increment the randomness
+        if now_ms <= last_ms:
             rand_int = int.from_bytes(_last[TIMESTAMP_LEN:], "big") + 1
             if rand_int >= 1 << (RANDOMNESS_LEN * 8):
                 raise OverflowError(
```