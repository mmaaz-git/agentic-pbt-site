import numpy as np
import sys

# Check what this value is
val = 2.22507386e-311

print(f"Value: {val}")
print(f"Is normal: {np.isfinite(val) and val != 0}")
print(f"Is denormalized: {val != 0 and abs(val) < sys.float_info.min}")
print(f"sys.float_info.min (smallest normal): {sys.float_info.min}")
print(f"np.finfo(float).tiny: {np.finfo(float).tiny}")
print(f"np.finfo(float).smallest_subnormal: {np.finfo(float).smallest_subnormal}")
print(f"Is val < tiny: {abs(val) < np.finfo(float).tiny}")

# Test dividing by this value
print(f"\n1.0 / val = {1.0 / val}")
print(f"Is inf: {np.isinf(1.0 / val)}")

# Check if it's a denormalized/subnormal number
print(f"\nValue in binary representation: {val.hex()}")
print(f"Smallest subnormal: {np.finfo(float).smallest_subnormal.hex()}")