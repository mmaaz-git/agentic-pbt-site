#!/usr/bin/env python3
import sys
sys.path.insert(0, '../../envs/llm_env/lib/python3.13/site-packages')

import time
import threading
from unittest.mock import patch
from llm.utils import monotonic_ulid, _last, _lock
import llm.utils

print("Testing monotonic_ulid behavior with simulated clock skew\n")

# Test that simulates clock going backwards
def test_with_mocked_time():
    # Reset the global state
    llm.utils._last = None

    # Mock time sequence: 1000ms, 1001ms, 999ms (clock goes backward)
    time_sequence = [1000_000_000, 1001_000_000, 999_000_000]  # nanoseconds
    time_iter = iter(time_sequence)

    def mock_time_ns():
        return next(time_iter)

    with patch('time.time_ns', side_effect=mock_time_ns):
        ulid1 = monotonic_ulid()
        print(f"ULID 1 (T=1000ms): {ulid1}")
        print(f"  Timestamp: {ulid1.milliseconds}ms")
        print(f"  Bytes: {ulid1.bytes.hex()}")

        ulid2 = monotonic_ulid()
        print(f"\nULID 2 (T=1001ms): {ulid2}")
        print(f"  Timestamp: {ulid2.milliseconds}ms")
        print(f"  Bytes: {ulid2.bytes.hex()}")

        ulid3 = monotonic_ulid()
        print(f"\nULID 3 (T=999ms, clock went backward!): {ulid3}")
        print(f"  Timestamp: {ulid3.milliseconds}ms")
        print(f"  Bytes: {ulid3.bytes.hex()}")

        print("\n=== Monotonicity Check ===")
        print(f"ulid2 > ulid1: {ulid2 > ulid1} ✓")
        print(f"ulid3 > ulid2: {ulid3 > ulid2} {'✓' if ulid3 > ulid2 else '✗ VIOLATED!'}")

        if ulid3 <= ulid2:
            print("\n❌ BUG CONFIRMED: Monotonicity violated when clock goes backward!")
            print(f"   ULID3 ({ulid3}) is not greater than ULID2 ({ulid2})")
            print(f"   This violates the documented guarantee of strict monotonicity")
            return False
        else:
            print("\n✓ Monotonicity maintained even with clock skew")
            return True

# Run the test
test_with_mocked_time()

# Additional analysis
print("\n=== Code Analysis ===")
print("The monotonic_ulid function has three branches:")
print("1. If _last is None (first call): Generate fresh ULID")
print("2. If now_ms == last_ms: Increment randomness")
print("3. Otherwise (implicit): Generate fresh ULID with current timestamp")
print("\nThe bug is in branch 3: When now_ms < last_ms (clock backward),")
print("it still generates a fresh ULID with the earlier timestamp,")
print("violating the strict monotonicity guarantee.")