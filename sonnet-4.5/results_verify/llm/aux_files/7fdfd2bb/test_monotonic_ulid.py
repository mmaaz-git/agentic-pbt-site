#!/usr/bin/env python3
"""Test script to reproduce the monotonic_ulid bug."""

import sys
import os
import time
import threading
from unittest.mock import patch

# Add the llm_env to the path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import monotonic_ulid, _fresh, _last, _lock, TIMESTAMP_LEN
from ulid import ULID

def test_normal_operation():
    """Test that monotonic_ulid works correctly under normal conditions."""
    print("Testing normal operation...")
    ulids = []
    for i in range(10):
        ulid = monotonic_ulid()
        if ulids:
            assert ulid > ulids[-1], f"ULID not strictly increasing: {ulid} <= {ulids[-1]}"
        ulids.append(ulid)
        if i < 3:
            print(f"  ULID {i+1}: {ulid}")
    print("✓ Normal operation test passed - all ULIDs are strictly increasing")
    return True

def test_clock_regression_simulated():
    """Simulate clock regression and test if monotonicity is violated."""
    print("\nTesting clock regression simulation...")

    # Generate first ULID
    ulid1 = monotonic_ulid()
    print(f"  First ULID: {ulid1}")

    # Get the timestamp from the last generated ULID
    import llm.utils
    with llm.utils._lock:
        last_ms = int.from_bytes(llm.utils._last[:TIMESTAMP_LEN], "big")
        print(f"  Timestamp in first ULID (ms): {last_ms}")

    # Simulate what would happen if the clock moved backward
    # We'll manually create a ULID with an earlier timestamp
    backward_ms = last_ms - 1000  # Go back 1 second
    print(f"  Simulated backward timestamp (ms): {backward_ms}")

    fake_ulid_bytes = _fresh(backward_ms)
    ulid2 = ULID(fake_ulid_bytes)

    print(f"  ULID with backward timestamp: {ulid2}")
    print(f"  Monotonicity violated? {ulid2 < ulid1} (should be True)")

    if ulid2 < ulid1:
        print("✓ Confirmed: Clock regression would violate monotonicity")
        return True
    else:
        print("✗ Unexpected: Clock regression did not violate monotonicity")
        return False

def test_clock_regression_with_mock():
    """Test clock regression using mocking to simulate backward time."""
    print("\nTesting clock regression with time mocking...")

    # Reset the global _last state for a clean test
    import llm.utils
    original_last = llm.utils._last
    llm.utils._last = None

    try:
        # Generate first ULID at time T
        with patch('time.time_ns') as mock_time:
            # Set initial time to 1000 seconds (in nanoseconds)
            mock_time.return_value = 1000 * 1_000_000_000
            ulid1 = monotonic_ulid()
            print(f"  First ULID (at time 1000s): {ulid1}")

            # Now simulate clock moving backward to 999 seconds
            mock_time.return_value = 999 * 1_000_000_000
            ulid2 = monotonic_ulid()
            print(f"  Second ULID (at time 999s): {ulid2}")

            # Check if monotonicity is violated
            monotonicity_violated = ulid2 < ulid1
            print(f"  Monotonicity violated? {monotonicity_violated} (should be True)")

            if monotonicity_violated:
                print("✓ Bug confirmed: Monotonicity is violated when clock moves backward")
                return True
            else:
                print("✗ Bug not reproduced: Monotonicity maintained despite clock regression")
                return False
    finally:
        # Restore original state
        llm.utils._last = original_last

def test_hypothesis_style():
    """Run a property-based test similar to the one in the bug report."""
    print("\nRunning property-based test (1000 iterations)...")
    ulids = []
    for i in range(1000):
        ulid = monotonic_ulid()
        if ulids:
            if not (ulid > ulids[-1]):
                print(f"✗ ULID not strictly increasing at iteration {i+1}: {ulid} <= {ulids[-1]}")
                return False
        ulids.append(ulid)
    print(f"✓ Property-based test passed - generated 1000 strictly increasing ULIDs")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing llm.utils.monotonic_ulid for clock regression bug")
    print("=" * 60)

    results = []

    # Test 1: Normal operation
    results.append(("Normal operation", test_normal_operation()))

    # Test 2: Property-based test
    results.append(("Property-based test", test_hypothesis_style()))

    # Test 3: Clock regression simulation
    results.append(("Clock regression simulation", test_clock_regression_simulated()))

    # Test 4: Clock regression with mocking
    results.append(("Clock regression with mock", test_clock_regression_with_mock()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(result[1] for result in results)

    if not all_passed:
        print("\nCONCLUSION: Bug is CONFIRMED - monotonic_ulid violates its")
        print("monotonicity guarantee when the system clock moves backward.")
    else:
        print("\nCONCLUSION: All tests passed under normal conditions.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)