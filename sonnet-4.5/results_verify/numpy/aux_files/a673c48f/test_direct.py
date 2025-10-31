import numpy as np
import numpy.ma as ma

# Test with np.float32 (type object)
print("Testing ma.default_fill_value(np.float32):")
try:
    result = ma.default_fill_value(np.float32)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with np.dtype(np.float32) (dtype instance) for comparison
print("\nTesting ma.default_fill_value(np.dtype(np.float32)):")
try:
    result = ma.default_fill_value(np.dtype(np.float32))
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with other type objects
print("\nTesting with other type objects:")
for dtype_type in [np.int64, np.float64, np.int32]:
    print(f"  Testing {dtype_type}:")
    try:
        result = ma.default_fill_value(dtype_type)
        print(f"    Result: {result}")
    except Exception as e:
        print(f"    Error: {type(e).__name__}: {e}")