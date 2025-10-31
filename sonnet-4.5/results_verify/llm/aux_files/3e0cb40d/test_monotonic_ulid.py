#!/usr/bin/env python3
"""Test script to reproduce the monotonic_ulid bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from llm.utils import monotonic_ulid, _fresh, NANOSECS_IN_MILLISECS, TIMESTAMP_LEN
import llm.utils
import time

# Test 1: Property-based test from the bug report
def test_monotonic_ulid_with_clock_regression():
    """Property-based test that demonstrates the bug when clock goes backwards."""
    print("\n=== Test 1: Property-based test ===")
    prev = monotonic_ulid()
    print(f"Initial ULID: {prev}")

    original_time_ns = time.time_ns
    for offset in [0, -1000000, -5000000]:
        time.time_ns = lambda: original_time_ns() + offset
        current = monotonic_ulid()
        print(f"With offset {offset}ns: {current}")
        try:
            assert current > prev, f"Monotonicity violated: {current} <= {prev}"
            print(f"  ✓ {current} > {prev}: True")
        except AssertionError as e:
            print(f"  ✗ Monotonicity violated: {current} <= {prev}")
            print(f"    FAILED: {e}")
        prev = current

    time.time_ns = original_time_ns
    print("Test 1 complete")


# Test 2: Specific reproduction case from the bug report
def test_specific_reproduction():
    """Specific reproduction case from the bug report."""
    print("\n=== Test 2: Specific reproduction ===")

    original_time_ns = time.time_ns
    fake_time = original_time_ns()

    # Reset the global _last state
    llm.utils._last = None

    # Generate first ULID with current time
    time.time_ns = lambda: fake_time
    ulid1 = monotonic_ulid()

    # Simulate clock going backwards by 5ms
    fake_time -= 5 * NANOSECS_IN_MILLISECS
    time.time_ns = lambda: fake_time
    ulid2 = monotonic_ulid()

    # Restore original time function
    time.time_ns = original_time_ns

    print(f"ULID 1: {ulid1}")
    print(f"ULID 2: {ulid2}")
    print(f"ULID 2 > ULID 1: {ulid2 > ulid1}")

    if ulid2 > ulid1:
        print("  ✓ Test passed: Monotonicity maintained")
    else:
        print("  ✗ Test failed: Monotonicity violated!")

    return ulid2 > ulid1


# Test 3: Additional test - clock goes backward multiple times
def test_multiple_clock_regressions():
    """Test multiple clock regressions."""
    print("\n=== Test 3: Multiple clock regressions ===")

    original_time_ns = time.time_ns
    fake_time = original_time_ns()

    # Reset the global _last state
    llm.utils._last = None

    ulids = []

    # Generate first ULID
    time.time_ns = lambda: fake_time
    ulid = monotonic_ulid()
    ulids.append(ulid)
    print(f"ULID 0 (time={fake_time}): {ulid}")

    # Clock goes backward by 10ms
    fake_time -= 10 * NANOSECS_IN_MILLISECS
    time.time_ns = lambda: fake_time
    ulid = monotonic_ulid()
    ulids.append(ulid)
    print(f"ULID 1 (time={fake_time}): {ulid}")

    # Clock goes backward by another 5ms
    fake_time -= 5 * NANOSECS_IN_MILLISECS
    time.time_ns = lambda: fake_time
    ulid = monotonic_ulid()
    ulids.append(ulid)
    print(f"ULID 2 (time={fake_time}): {ulid}")

    # Clock goes forward by 20ms
    fake_time += 20 * NANOSECS_IN_MILLISECS
    time.time_ns = lambda: fake_time
    ulid = monotonic_ulid()
    ulids.append(ulid)
    print(f"ULID 3 (time={fake_time}): {ulid}")

    # Restore original time function
    time.time_ns = original_time_ns

    # Check monotonicity
    all_monotonic = True
    for i in range(1, len(ulids)):
        is_monotonic = ulids[i] > ulids[i-1]
        print(f"  ULID {i} > ULID {i-1}: {is_monotonic}")
        if not is_monotonic:
            all_monotonic = False
            print(f"    ✗ Monotonicity violated!")

    if all_monotonic:
        print("  ✓ All ULIDs are strictly monotonic")
    else:
        print("  ✗ Monotonicity violations detected")

    return all_monotonic


if __name__ == "__main__":
    print("Testing monotonic_ulid with clock regressions")
    print("=" * 50)

    try:
        test_monotonic_ulid_with_clock_regression()
    except Exception as e:
        print(f"Test 1 error: {e}")

    try:
        test_specific_reproduction()
    except Exception as e:
        print(f"Test 2 error: {e}")

    try:
        test_multiple_clock_regressions()
    except Exception as e:
        print(f"Test 3 error: {e}")

    print("\n" + "=" * 50)
    print("Testing complete")