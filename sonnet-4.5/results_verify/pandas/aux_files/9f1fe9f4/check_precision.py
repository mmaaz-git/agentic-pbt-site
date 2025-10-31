#!/usr/bin/env python3
import sys
import numpy as np

# Check what 5e-324 represents
print(f"5e-324 = {5e-324}")
print(f"Is it the smallest positive float? {5e-324 == np.nextafter(0.0, 1.0)}")
print(f"sys.float_info.min = {sys.float_info.min}")
print(f"Smallest positive normal float = {sys.float_info.min}")
print(f"Smallest positive subnormal float = {np.nextafter(0.0, 1.0)}")
print(f"5e-324 in hex: {float.hex(5e-324)}")

# Check floating point arithmetic
x = 5e-324
print(f"\n--- Floating point arithmetic with {x} ---")
print(f"x = {x}")
print(f"x * 0.001 = {x * 0.001}")
print(f"x - (x * 0.001) = {x - (x * 0.001)}")
print(f"Are they equal? {x == x - (x * 0.001)}")

# Explore near the boundary
print("\n--- Testing linspace with tiny ranges ---")
for exp in range(-324, -300, 5):
    val = 10 ** exp
    bins = np.linspace(0, val, 3)
    unique = np.unique(bins)
    print(f"10^{exp}: linspace gives {len(unique)} unique values out of 3")