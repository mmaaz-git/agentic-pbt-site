import sys
import numpy as np
import math

# Check the denormalized float
val = -2.225073858507e-311
print(f"Value: {val}")
print(f"Is finite: {np.isfinite(val)}")
print(f"Hex representation: {val.hex()}")

# Check the smallest normal float
print(f"\nSmallest normal float: {sys.float_info.min}")
print(f"Is val < smallest normal: {abs(val) < sys.float_info.min}")

# Check what happens with pd.cut
import pandas as pd

print("\n--- Testing pd.cut with denormalized floats ---")
data = [0.0, 0.0, 0.0, 0.0, -2.225073858507e-311]
print(f"Data range: min={min(data)}, max={max(data)}")
print(f"Range: {max(data) - min(data)}")

# Try with explicit bins
print("\n--- Testing with explicit bins ---")
try:
    result = pd.cut(data, bins=[-1e-300, 0, 1e-300])
    print(f"Success with explicit bins: {result}")
except Exception as e:
    print(f"Error with explicit bins: {e}")

# Test with larger range
print("\n--- Testing with normal data ---")
normal_data = [0.0, 0.0, 0.0, 0.0, 1.0]
try:
    result = pd.cut(normal_data, bins=2)
    print(f"Success with normal data: works fine")
except Exception as e:
    print(f"Error with normal data: {e}")