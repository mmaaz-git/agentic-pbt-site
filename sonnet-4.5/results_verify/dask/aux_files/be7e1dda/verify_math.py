#!/usr/bin/env python3
"""Verify the mathematical calculation"""

# The failing value
n = 1125894277343089729

# Check the calculation
PiB = 2**50
print(f"n = {n}")
print(f"PiB = 2**50 = {PiB}")
print(f"n / PiB = {n / PiB:.6f}")
print(f"Formatted as .2f: {n / PiB:.2f}")

# Check the threshold condition
threshold = PiB * 0.9
print(f"\nThreshold (PiB * 0.9) = {threshold:.0f}")
print(f"n >= threshold? {n >= threshold}")

# The function's logic:
# if n >= k * 0.9:
#     return f"{n / k:.2f} {prefix}B"

# So for this value, it should format as PiB
result = f"{n / PiB:.2f} PiB"
print(f"\nResult: '{result}'")
print(f"Length: {len(result)}")

# Check other boundary cases
print("\nOther interesting values:")
for mult in [999.99, 1000.00, 1023.99, 1024.00]:
    val = int(PiB * mult)
    if val < 2**60:
        res = f"{val / PiB:.2f} PiB"
        print(f"  {mult:7.2f} PiB -> '{res}' (len={len(res)})")