import sys
import math

print("Python float limits:")
print(f"  sys.float_info.max = {sys.float_info.max}")
print(f"  sys.float_info.min = {sys.float_info.min}")
print(f"  In scientific notation: {sys.float_info.max:.5e}")

print("\nExcel's documented max according to search: 1.79769313486232E308")
print(f"Are they the same? {abs(sys.float_info.max - 1.79769313486232E308) < 1e305}")

print("\nTest value from bug report: 1.7976931348623155e+308")
print(f"Is it less than sys.float_info.max? {1.7976931348623155e+308 < sys.float_info.max}")
print(f"Difference: {sys.float_info.max - 1.7976931348623155e+308}")

# Let's see what happens with values near the limit
test_values = [
    sys.float_info.max,
    sys.float_info.max * 0.9999,
    sys.float_info.max * 1.0001,
    1.7976931348623155e+308,
    float('inf')
]

print("\nTest values and their properties:")
for val in test_values:
    print(f"  Value: {val:.5e}")
    print(f"    isfinite: {math.isfinite(val)}")
    print(f"    isinf: {math.isinf(val)}")
    try:
        int_val = int(val)
        print(f"    int(val): {int_val}")
    except OverflowError as e:
        print(f"    int(val): OverflowError - {e}")