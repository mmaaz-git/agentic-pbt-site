#!/usr/bin/env python3
"""Test script to understand the root cause of the bug"""

import sys
import numpy as np
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.io.formats.format import get_precision

print("Testing get_precision function with edge cases:")
print("=" * 60)

# Test case 1: [0.0, 5e-324]
array1 = np.array([0.0, 5e-324]) * 100
print(f"\nInput: {[0.0, 5e-324]} (after *100: {array1})")
print(f"Min value: {np.min(array1)}, Max value: {np.max(array1)}")

# Calculate differences
to_begin = array1[0] if array1[0] > 0 else None
to_end = 100 - array1[-1] if array1[-1] < 100 else None
diff = np.ediff1d(array1, to_begin=to_begin, to_end=to_end)
diff = abs(diff)
print(f"Differences: {diff}")
print(f"Min diff: {np.min(diff)}")

# Try to calculate log10
try:
    log_val = np.log10(np.min(diff))
    print(f"log10(min_diff): {log_val}")
    prec = -np.floor(log_val).astype(int)
    print(f"Precision calculated: {prec}")
except Exception as e:
    print(f"Error calculating precision: {e}")

# Try get_precision
try:
    prec = get_precision(array1)
    print(f"get_precision result: {prec}")
except Exception as e:
    print(f"get_precision error: {e}")

# Test what happens with large precision
print("\n" + "-" * 40)
print("Testing round() with large precision values:")
test_val = np.array([0.0, 5e-324]) * 100
for prec_test in [10, 50, 100, 200, 300, 320, 330]:
    try:
        rounded = test_val.round(prec_test)
        print(f"Precision {prec_test}: {rounded}")
    except Exception as e:
        print(f"Precision {prec_test}: ERROR - {e}")

# Test case 2: [0.0, 1.401298464324817e-45]
print("\n" + "=" * 60)
array2 = np.array([0.0, 1.401298464324817e-45]) * 100
print(f"\nInput: {[0.0, 1.401298464324817e-45]} (after *100: {array2})")
try:
    prec2 = get_precision(array2)
    print(f"get_precision result: {prec2}")
    rounded2 = array2.round(prec2)
    print(f"Rounded values: {rounded2}")
    print(f"As int: {rounded2.astype(int)}")
except Exception as e:
    print(f"Error: {e}")

# Test case 3: [1e-10]
print("\n" + "=" * 60)
array3 = np.array([1e-10]) * 100
print(f"\nInput: {[1e-10]} (after *100: {array3})")
try:
    prec3 = get_precision(array3)
    print(f"get_precision result: {prec3}")
    rounded3 = array3.round(prec3)
    print(f"Rounded values: {rounded3}")
    print(f"As string: {rounded3.astype(str)}")
except Exception as e:
    print(f"Error: {e}")

# Show Python/NumPy float precision limits
print("\n" + "=" * 60)
print("Float precision information:")
print(f"Smallest positive normal float: {np.finfo(float).tiny}")
print(f"Smallest positive subnormal float: {np.nextafter(0.0, 1.0)}")
print(f"Machine epsilon: {np.finfo(float).eps}")
print(f"Max decimals for float64: ~15-17 digits")