#!/usr/bin/env python3
"""Minimal demonstration of format_bytes bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from dask.utils import format_bytes

# The specific failing input from Hypothesis
n = 1125894277343089729

# Call the function
result = format_bytes(n)

# Display the results
print(f"Input value: {n}")
print(f"Is n < 2^60? {n < 2**60}")
print(f"Output: '{result}'")
print(f"Output length: {len(result)} characters")
print(f"Expected: <= 10 characters (per documentation)")
print()

# Verify the violation
assert n < 2**60, f"Value {n} is not less than 2^60"
assert len(result) == 11, f"Expected output length 11, got {len(result)}"
assert result == '1000.00 PiB', f"Expected '1000.00 PiB', got '{result}'"

print("Bug confirmed: Output exceeds documented 10-character limit")