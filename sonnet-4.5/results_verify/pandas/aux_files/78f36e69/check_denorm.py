import sys
import numpy as np

# Check the denormalized float
val = -2.225073858507e-311
print(f"Value: {val}")
print(f"Is finite: {np.isfinite(val)}")
print(f"Is normal: {np.isnormal(val)}")
print(f"Is denormalized: {not np.isnormal(val) and val != 0}")
print(f"Hex representation: {val.hex()}")

# Check the smallest normal float
print(f"\nSmallest normal float: {sys.float_info.min}")
print(f"Smallest subnormal float: {sys.float_info.min * sys.float_info.epsilon}")

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