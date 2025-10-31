import numpy as np
from xarray.core.indexes import normalize_label

values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

print("Testing normalize_label with np.float32 (type)...")
try:
    result = normalize_label(values, dtype=np.float32)
    print(f"Success! Result dtype: {result.dtype}")
except AttributeError as e:
    print(f"AttributeError: {e}")

print("\nTesting normalize_label with np.dtype('float32') (dtype instance)...")
try:
    result = normalize_label(values, dtype=np.dtype('float32'))
    print(f"Success! Result dtype: {result.dtype}")
except AttributeError as e:
    print(f"AttributeError: {e}")

# Test that numpy itself accepts both forms
print("\nTesting numpy's behavior with dtype types vs instances...")
arr1 = np.array([1,2,3], dtype=np.float32)
arr2 = np.array([1,2,3], dtype=np.dtype('float32'))
print(f"np.array with np.float32 dtype: {arr1.dtype}")
print(f"np.array with np.dtype('float32') dtype: {arr2.dtype}")