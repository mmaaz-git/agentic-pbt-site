# Bug Report: llm.utils.monotonic_ulid - Violates Strict Monotonicity When System Clock Regresses

**Target**: `llm.utils.monotonic_ulid`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `monotonic_ulid` function violates its documented guarantee of strict monotonicity when the system clock moves backwards, generating a new ULID with an earlier timestamp rather than maintaining ordering by incrementing the randomness component.

## Property-Based Test

```python
#!/usr/bin/env python3
from hypothesis import given, strategies as st, settings
from llm.utils import monotonic_ulid, NANOSECS_IN_MILLISECS
import llm.utils
import time

@given(clock_regress_ms=st.integers(1, 100))
@settings(max_examples=100)
def test_monotonic_ulid_clock_regression(clock_regress_ms):
    """Test that monotonic_ulid violates monotonicity when clock goes backwards."""
    # Store original time function
    original_time_ns = time.time_ns
    base_time = original_time_ns()

    # Reset global state
    llm.utils._last = None

    # Generate first ULID at base time
    time.time_ns = lambda: base_time
    ulid1 = monotonic_ulid()

    # Simulate clock going backward
    regressed_time = base_time - (clock_regress_ms * NANOSECS_IN_MILLISECS)
    time.time_ns = lambda: regressed_time
    ulid2 = monotonic_ulid()

    # Restore original time function
    time.time_ns = original_time_ns

    # Check if monotonicity is preserved (this should fail)
    assert ulid2 > ulid1, f"Monotonicity violated when clock went back {clock_regress_ms}ms: {ulid2} <= {ulid1}"

if __name__ == "__main__":
    # Run the hypothesis test
    test_monotonic_ulid_clock_regression()
```

<details>

<summary>
**Failing input**: `clock_regress_ms=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 35, in <module>
    test_monotonic_ulid_clock_regression()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 8, in test_monotonic_ulid_clock_regression
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 31, in test_monotonic_ulid_clock_regression
    assert ulid2 > ulid1, f"Monotonicity violated when clock went back {clock_regress_ms}ms: {ulid2} <= {ulid1}"
           ^^^^^^^^^^^^^
AssertionError: Monotonicity violated when clock went back 1ms: 01K61B5V66X4QH467CX4QC1F9Y <= 01K61B5V67CBKXTHETG3PQ7Z21
Falsifying example: test_monotonic_ulid_clock_regression(
    clock_regress_ms=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
from llm.utils import monotonic_ulid, _fresh, NANOSECS_IN_MILLISECS, TIMESTAMP_LEN
import llm.utils
import time

# Store the original time function
original_time_ns = time.time_ns
fake_time = original_time_ns()

# Reset the global state
llm.utils._last = None

# Set initial time
time.time_ns = lambda: fake_time
ulid1 = monotonic_ulid()
print(f"ULID 1: {ulid1}")

# Simulate clock going backward by 5 milliseconds
fake_time -= 5 * NANOSECS_IN_MILLISECS
time.time_ns = lambda: fake_time
ulid2 = monotonic_ulid()
print(f"ULID 2: {ulid2}")

# Restore original time function
time.time_ns = original_time_ns

# Check if monotonicity is preserved
print(f"ULID 2 > ULID 1: {ulid2 > ulid1}")
print(f"Monotonicity violated: {ulid2 <= ulid1}")
```

<details>

<summary>
Clock regression causes ULID ordering violation
</summary>
```
ULID 1: 01K61B42431S4WBV70KA23WXNV
ULID 2: 01K61B423YZQEZYM2ESWSHG6TW
ULID 2 > ULID 1: False
Monotonicity violated: True
```
</details>

## Why This Is A Bug

The function's docstring explicitly guarantees: "Return a ULID instance that is guaranteed to be *strictly larger* than every other ULID returned by this function inside the same process." This guarantee is unconditional and does not exclude clock regression scenarios.

The implementation fails to uphold this guarantee because the condition at line 716 only checks for timestamp equality (`if now_ms == last_ms:`), not for timestamps that have gone backward. When `now_ms < last_ms` due to clock regression, the code falls through to line 728, which generates a fresh ULID with the earlier timestamp, resulting in a lexicographically smaller ULID that violates strict monotonicity.

Clock regression is a documented real-world phenomenon that occurs in production systems due to:
- NTP (Network Time Protocol) synchronization adjustments
- Virtual machine migrations or suspensions
- Manual system time adjustments
- Leap second adjustments
- Hardware clock drift corrections

Furthermore, the function claims to work "the same way the reference JavaScript `monotonicFactory` does", but the JavaScript implementation preserves monotonicity even when timestamps go backward, making this Python implementation incompatible with its stated reference.

## Relevant Context

The bug is in `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py` at lines 715-729. The problematic logic is:

```python
# Line 716: Only checks for equality, not regression
if now_ms == last_ms:
    # Increment randomness logic
    return ULID(_last)

# Line 728: Falls through here when now_ms < last_ms, violating monotonicity
_last = _fresh(now_ms)
return ULID(_last)
```

The ULID specification (https://github.com/ulid/spec) defines ULIDs as 128-bit identifiers with:
- 48-bit timestamp (milliseconds since Unix epoch)
- 80-bit randomness

The lexicographic ordering of ULIDs depends first on the timestamp component, so generating a ULID with an earlier timestamp will always produce a smaller ULID, regardless of the randomness component.

The JavaScript reference implementation (https://github.com/ulid/javascript) explicitly handles this case to maintain monotonicity even with backward timestamps, as stated in their documentation: "Even if a lower timestamp is passed (or generated), it will preserve sort order."

## Proposed Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -713,8 +713,8 @@ def monotonic_ulid() -> ULID:
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