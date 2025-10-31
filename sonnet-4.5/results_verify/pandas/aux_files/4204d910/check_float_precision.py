import numpy as np
import sys

print("Floating-point precision limits:")
print("=" * 60)

# IEEE 754 double precision info
print(f"sys.float_info.min: {sys.float_info.min}")
print(f"sys.float_info.max: {sys.float_info.max}")
print(f"sys.float_info.epsilon: {sys.float_info.epsilon}")
print(f"sys.float_info.dig: {sys.float_info.dig}")  # decimal digits of precision
print(f"sys.float_info.mant_dig: {sys.float_info.mant_dig}")  # mantissa digits in base 2

print("\nNumPy float64 info:")
print(f"np.finfo(np.float64).min: {np.finfo(np.float64).min}")
print(f"np.finfo(np.float64).max: {np.finfo(np.float64).max}")
print(f"np.finfo(np.float64).precision: {np.finfo(np.float64).precision}")

# Test the extreme value
extreme_val = 1.1125369292536007e-308
print(f"\nExtreme value: {extreme_val}")
print(f"Is normal? {np.isfinite(extreme_val) and extreme_val != 0}")
print(f"Is subnormal? {abs(extreme_val) < sys.float_info.min and extreme_val != 0}")

# Check if numpy has documentation on around limits
print("\nTesting numpy.around() behavior with different decimal values:")
for decimals in [10, 15, 20, 50, 100, 200, 300, 310, 320]:
    try:
        result = np.around(extreme_val, decimals)
        print(f"  decimals={decimals:3d}: result={result}, isnan={np.isnan(result)}")
    except Exception as e:
        print(f"  decimals={decimals:3d}: Error: {e}")

# Float64 has approximately 15-17 decimal digits of precision
print("\nNote: float64 has about 15-17 decimal digits of precision")
print("Asking numpy.around to round to 310 decimal places exceeds this by ~20x")