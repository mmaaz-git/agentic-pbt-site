import pandas as pd
import traceback

print("Testing direct example from bug report:")
print("=" * 50)

values = [2.2250738585e-313, -1.0]
print(f"Input values: {values}")
print(f"bins: 2")
print()

try:
    result = pd.cut(values, bins=2)
    print(f"Success! Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()

print("\nAdditional information:")
print(f"Value types: {[type(v).__name__ for v in values]}")
print(f"First value is denormal: {2.2250738585e-313}")
print(f"Smallest normal float: {2.2250738585072014e-308}")
print(f"Is denormal? {2.2250738585e-313 < 2.2250738585072014e-308}")

# Let's also check what happens with the values during processing
import numpy as np
print("\nNumpy array conversion:")
arr = np.array(values)
print(f"Array: {arr}")
print(f"Array dtype: {arr.dtype}")
print(f"Min: {np.min(arr)}")
print(f"Max: {np.max(arr)}")
print(f"Range: {np.max(arr) - np.min(arr)}")