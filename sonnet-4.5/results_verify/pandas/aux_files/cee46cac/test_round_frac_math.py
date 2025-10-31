#!/usr/bin/env python3
"""Verify the math of why _round_frac produces NaN for very small floats"""

import numpy as np

def analyze_round_frac(x, precision=3):
    """Simulate the _round_frac function logic"""
    print(f"\nAnalyzing _round_frac for x = {x:.6e}, precision = {precision}")
    print("-" * 60)

    if not np.isfinite(x) or x == 0:
        print(f"Early return: x = {x}")
        return x

    frac, whole = np.modf(x)
    print(f"np.modf({x:.6e}) = (frac={frac:.6e}, whole={whole:.6e})")

    if whole == 0:
        log_val = np.log10(abs(frac))
        floor_log = np.floor(log_val)
        digits = -int(floor_log) - 1 + precision

        print(f"whole == 0, so calculating digits:")
        print(f"  log10(abs({frac:.6e})) = {log_val}")
        print(f"  floor(log10(abs(frac))) = {floor_log}")
        print(f"  digits = -int({floor_log}) - 1 + {precision} = {digits}")
    else:
        digits = precision
        print(f"whole != 0, so digits = precision = {digits}")

    result = np.around(x, digits)
    print(f"np.around({x:.6e}, {digits}) = {result}")

    if not np.isfinite(result):
        print(f"WARNING: Result is {result} (not finite!)")
        print(f"This happens when digits={digits} is too large for np.around()")

    return result

# Test very small numbers
test_values = [
    2.2250738585e-313,  # The problematic value from the bug report
    2.2250738585072014e-308,  # Another problematic value
    1e-100,  # Medium-small value
    1e-10,   # Small but reasonable value
    0.001,   # Normal small value
]

for val in test_values:
    analyze_round_frac(val, precision=3)

# Show what happens with extreme digit values
print("\n" + "="*60)
print("Testing np.around() with extreme digit values:")
print("="*60)

x = 2.2250738585e-313
for digits in [10, 50, 100, 200, 300, 310, 312, 313, 314, 315]:
    result = np.around(x, digits)
    print(f"np.around({x:.3e}, {digits:3d}) = {result}")

# Check the maximum safe digits for np.around
print("\n" + "="*60)
print("Finding maximum safe digits for np.around():")
print("="*60)

x = 2.2250738585e-313
for digits in range(300, 320):
    result = np.around(x, digits)
    if not np.isfinite(result):
        print(f"np.around() produces NaN starting at digits={digits}")
        break
else:
    print("All tested digits produced finite results")