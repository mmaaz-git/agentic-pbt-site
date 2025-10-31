#!/usr/bin/env python3
"""Test to find exact boundary where bug occurs"""

from dask.utils import format_bytes

# Let's find the exact threshold where it breaks
print("--- Finding the exact threshold ---")
for multiplier in [999, 999.9, 999.99, 1000, 1001]:
    val = int(multiplier * 2**50)
    res = format_bytes(val)
    print(f"{multiplier:7.2f} * 2^50 = {val:20} -> '{res:11}' (len={len(res)})")

# Verify the math on when this breaks
print("\n--- Understanding when the bug occurs ---")
print("The function uses n / k where k = 2**50 for PiB")
print("It formats with .2f, which means 2 decimal places")
print("When n / 2**50 >= 1000, we get 4 or more digits before the decimal")
print(f"1000 * 2**50 = {1000 * 2**50}")
print(f"This is still < 2**60 = {2**60}")
print(f"In fact, 1000 * 2**50 / 2**60 = {(1000 * 2**50) / 2**60:.2%} of 2**60")

# Let's also check what happens at even larger values within 2**60
print("\n--- Testing larger values still within 2**60 ---")
test_vals = [
    1000 * 2**50,
    1024 * 2**50,  # This would be exactly 1 EiB (exbibyte)
    2000 * 2**50,
    int(0.99 * 2**60),  # 99% of the limit
]

for val in test_vals:
    if val < 2**60:
        res = format_bytes(val)
        print(f"{val:22} -> '{res:11}' (len={len(res)}, {val/2**60:.1%} of 2^60)")