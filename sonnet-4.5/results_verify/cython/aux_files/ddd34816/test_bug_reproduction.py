#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import time
from unittest import mock
from llm.utils import monotonic_ulid, NANOSECS_IN_MILLISECS

print("=== Reproducing the bug ===\n")

# Generate first ULID
ulid1 = monotonic_ulid()
print(f"First ULID: {ulid1}")

# Get current time and simulate backward clock
current_time_ns = time.time_ns()
backward_time_ns = current_time_ns - (2 * NANOSECS_IN_MILLISECS)

# Generate second ULID with backward clock
with mock.patch('time.time_ns', return_value=backward_time_ns):
    ulid2 = monotonic_ulid()
    print(f"Second ULID (after clock went backward): {ulid2}")

    print(f"\nComparison results:")
    print(f"ulid1 < ulid2: {ulid1 < ulid2}")
    print(f"ulid1 >= ulid2: {ulid1 >= ulid2}")

    if ulid1 >= ulid2:
        print("\n*** BUG CONFIRMED: Monotonicity violated! ***")
        print("The second ULID is not strictly larger than the first ULID")
    else:
        print("\nNo bug: Monotonicity maintained")