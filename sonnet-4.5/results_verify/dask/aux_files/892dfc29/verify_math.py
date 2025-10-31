#!/usr/bin/env python3
"""Verify the mathematical claims in the bug report."""

# Check the mathematical claims
print("Mathematical verification:")
print("-" * 50)

# Check that 1000 * 2**50 < 2**60
value = 1000 * 2**50
limit = 2**60

print(f"1000 * 2**50 = {value}")
print(f"2**60 = {limit}")
print(f"1000 * 2**50 < 2**60: {value < limit}")
print()

# Check how much room we have
ratio = value / limit
print(f"1000 * 2**50 is {ratio:.4%} of 2**60")
print(f"Maximum multiplier for 2**50 that stays under 2**60: {2**60 // 2**50} = {2**10} = 1024")
print()

# So values from 1000 * 2**50 to 1024 * 2**50 - 1 are:
# - Less than 2**60 (satisfying the docstring condition)
# - Will format as >= 1000.00 PiB (11+ characters)

print("Range of problematic values:")
print(f"From: {1000 * 2**50} (1000 * 2**50)")
print(f"To:   {2**60 - 1} (2**60 - 1)")
print()

# Test the format for these edge cases
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')
import dask.utils

print("Formatting edge cases:")
for n in [1000 * 2**50, 1023 * 2**50, 1024 * 2**50 - 1, 2**60 - 1, 2**60]:
    if n < 2**60:
        result = dask.utils.format_bytes(n)
        print(f"format_bytes({n:25}) = '{result:12}' (len={len(result)}) [< 2**60]")
    else:
        result = dask.utils.format_bytes(n)
        print(f"format_bytes({n:25}) = '{result:12}' (len={len(result)}) [>= 2**60]")