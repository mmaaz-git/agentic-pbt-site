# Bug Report: llm.utils.monotonic_ulid Monotonicity Violation on Clock Regression

**Target**: `llm.utils.monotonic_ulid`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `monotonic_ulid` function violates its monotonicity guarantee when the system clock moves backward, generating ULIDs that are smaller than previously generated ones despite promising "strictly larger" ULIDs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from llm.utils import monotonic_ulid
import time
from unittest.mock import patch, MagicMock

def test_monotonic_ulid_always_increasing():
    """Test that monotonic_ulid always returns strictly increasing ULIDs"""
    ulids = []
    for _ in range(100):
        ulid = monotonic_ulid()
        if ulids:
            assert ulid > ulids[-1], f"ULID not strictly increasing: {ulid} <= {ulids[-1]}"
        ulids.append(ulid)
    print("✓ Normal operation: All ULIDs are strictly increasing")

# Test with simulated clock regression
@given(st.integers(min_value=1, max_value=10000))
@settings(max_examples=10, deadline=1000)
def test_monotonic_ulid_clock_regression(regression_ms):
    """Test monotonic_ulid behavior when clock moves backward"""

    # Mock time to control clock
    with patch('llm.utils.time') as mock_time:
        # Start at time 1000 seconds
        initial_time_ns = 1000 * 1_000_000_000
        mock_time.time_ns = MagicMock(return_value=initial_time_ns)

        # Reset global state for clean test
        import llm.utils
        llm.utils._last = None

        # Generate first ULID at time 1000s
        ulid1 = monotonic_ulid()

        # Move clock backward
        backward_time_ns = (1000 * 1_000_000_000) - (regression_ms * 1_000_000)
        mock_time.time_ns = MagicMock(return_value=backward_time_ns)

        # Generate second ULID after clock regression
        ulid2 = monotonic_ulid()

        # This SHOULD still be monotonically increasing per the function's contract
        assert ulid2 > ulid1, f"Monotonicity violated on {regression_ms}ms clock regression: {ulid2} <= {ulid1}"

if __name__ == "__main__":
    print("Running hypothesis tests for monotonic_ulid...")
    print("\n1. Testing normal operation (no clock regression):")
    test_monotonic_ulid_always_increasing()

    print("\n2. Testing with clock regression (property-based test):")
    try:
        test_monotonic_ulid_clock_regression()
        print("✓ All clock regression tests passed (unexpected - bug may be fixed)")
    except AssertionError as e:
        print(f"✗ Clock regression test FAILED as expected:")
        print(f"  {e}")
        print("\n  This confirms the bug: monotonic_ulid violates its monotonicity")
        print("  guarantee when the system clock moves backward.")
```

<details>

<summary>
**Failing input**: `Clock regression of 1ms`
</summary>
```
Running hypothesis tests for monotonic_ulid...

1. Testing normal operation (no clock regression):
✓ Normal operation: All ULIDs are strictly increasing

2. Testing with clock regression (property-based test):
✗ Clock regression test FAILED as expected:
  Monotonicity violated on 1ms clock regression: 000000YGHZH4GZAEG16964R2JP <= 000000YGJ025938Z1RRR45JSSZ

  This confirms the bug: monotonic_ulid violates its monotonicity
  guarantee when the system clock moves backward.
```
</details>

## Reproducing the Bug

```python
from llm.utils import monotonic_ulid, _fresh, _lock, TIMESTAMP_LEN
from ulid import ULID
import llm.utils
import time

# Generate first ULID normally
ulid1 = monotonic_ulid()
print(f"First ULID: {ulid1}")

# Access the last timestamp from the module-level _last variable
with _lock:
    last_bytes = llm.utils._last
    if last_bytes:
        last_ms = int.from_bytes(last_bytes[:TIMESTAMP_LEN], "big")
        print(f"Timestamp in first ULID: {last_ms} ms")

        # Simulate clock regression by generating a ULID 1 second earlier
        backward_ms = last_ms - 1000
        print(f"Simulated backward timestamp: {backward_ms} ms (1 second earlier)")

        # Create a ULID with the earlier timestamp
        fake_ulid_bytes = _fresh(backward_ms)
        ulid2 = ULID(fake_ulid_bytes)

        print(f"\nULID after clock regression: {ulid2}")
        print(f"Comparison: ulid2 < ulid1 = {ulid2 < ulid1}")
        print(f"This violates monotonicity! The function promises strictly increasing ULIDs.")
        print(f"\nExpected: All ULIDs should be strictly larger than previous ones")
        print(f"Actual: ulid2 ({ulid2}) < ulid1 ({ulid1})")
    else:
        print("Error: _last is None")
```

<details>

<summary>
Demonstration of monotonicity violation
</summary>
```
First ULID: 01K61B5BTKNMJE9VZ77YNJZ1ME
Timestamp in first ULID: 1758834372435 ms
Simulated backward timestamp: 1758834371435 ms (1 second earlier)

ULID after clock regression: 01K61B5AVBH0DF1YEH66K78DMR
Comparison: ulid2 < ulid1 = True
This violates monotonicity! The function promises strictly increasing ULIDs.

Expected: All ULIDs should be strictly larger than previous ones
Actual: ulid2 (01K61B5AVBH0DF1YEH66K78DMR) < ulid1 (01K61B5BTKNMJE9VZ77YNJZ1ME)
```
</details>

## Why This Is A Bug

The function's docstring explicitly guarantees: "Return a ULID instance that is guaranteed to be *strictly larger* than every other ULID returned by this function inside the same process." This guarantee is violated when the system clock moves backward.

The bug occurs because:
1. The function gets current time: `now_ms = time.time_ns() // NANOSECS_IN_MILLISECS`
2. When `now_ms < last_ms` (clock moved backward), there's no special handling
3. The code path falls through to generating a fresh ULID with the earlier timestamp
4. Since ULIDs are lexicographically ordered with timestamp as the most significant 48 bits, the new ULID is smaller
5. This directly violates the documented "strictly larger" guarantee

The function claims to work "the same way the reference JavaScript `monotonicFactory` does", but the JavaScript implementation actually handles clock regression properly by preserving monotonicity even when the clock moves backward.

## Relevant Context

System clocks can move backward in production due to:
- **NTP synchronization** - Network Time Protocol adjustments are common in production systems
- **Virtual machine migrations** - VMs can experience clock jumps when migrated between hosts
- **Manual clock adjustments** - System administrators may manually adjust time
- **Hardware clock drift corrections** - Automatic corrections for hardware clock inaccuracies
- **Leap second handling** - Rare but can cause clock adjustments
- **System suspend/resume cycles** - Can cause apparent time regression

The ULID specification (https://github.com/ulid/spec) doesn't explicitly mandate clock regression handling, but the function's documentation makes an absolute promise of monotonicity without exceptions.

Code location: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py` lines 689-736

## Proposed Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -712,6 +712,11 @@ def monotonic_ulid() -> ULID:
         # Decode timestamp from the last ULID we handed out
         last_ms = int.from_bytes(_last[:TIMESTAMP_LEN], "big")

+        # Handle clock regression: treat it as same millisecond to maintain monotonicity
+        if now_ms < last_ms:
+            # Clock moved backward, use last_ms to maintain monotonicity
+            now_ms = last_ms
+
         # If the millisecond is the same, increment the randomness
         if now_ms == last_ms:
             rand_int = int.from_bytes(_last[TIMESTAMP_LEN:], "big") + 1
```