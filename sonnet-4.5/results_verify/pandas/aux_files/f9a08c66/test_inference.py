from pandas.core.dtypes.inference import is_integer, is_float
import numpy as np

# Test is_integer
print("Testing is_integer:")
print(f"is_integer(5): {is_integer(5)}")
print(f"is_integer(5.0): {is_integer(5.0)}")
print(f"is_integer(np.int32(5)): {is_integer(np.int32(5))}")
print(f"is_integer(np.int64(5)): {is_integer(np.int64(5))}")
print(f"is_integer('5'): {is_integer('5')}")

print("\nTesting is_float:")
print(f"is_float(5): {is_float(5)}")
print(f"is_float(5.0): {is_float(5.0)}")
print(f"is_float(5.5): {is_float(5.5)}")
print(f"is_float(np.float32(5.0)): {is_float(np.float32(5.0))}")
print(f"is_float(np.float64(5.0)): {is_float(np.float64(5.0))}")
print(f"is_float('5.0'): {is_float('5.0')}")