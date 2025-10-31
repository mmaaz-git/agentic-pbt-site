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