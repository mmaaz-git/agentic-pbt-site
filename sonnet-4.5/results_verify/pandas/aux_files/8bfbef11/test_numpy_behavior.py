import numpy as np
import pyarrow as pa

print("Testing underlying implementations with empty indices:")

print("\n1. NumPy array.take():")
np_array = np.array([1, 2, 3])
result = np_array.take([])
print(f"   np.array([1,2,3]).take([]): {result}, shape: {result.shape}, dtype: {result.dtype}")

print("\n2. PyArrow array.take() with proper indices:")
pa_array = pa.array([1, 2, 3])
try:
    # Try with empty integer array
    indices = pa.array([], type=pa.int64())
    result = pa_array.take(indices)
    print(f"   pa.array([1,2,3]).take(pa.array([], type=pa.int64())): {result}")
except Exception as e:
    print(f"   ERROR with int64 indices: {e}")

try:
    # Try with np.array([], dtype=int)
    indices = np.array([], dtype=int)
    result = pa_array.take(indices)
    print(f"   pa.array([1,2,3]).take(np.array([], dtype=int)): {result}")
except Exception as e:
    print(f"   ERROR with numpy int indices: {e}")

try:
    # Try with np.array([], dtype=float) - this is what pandas gives us
    indices = np.array([], dtype=float)
    result = pa_array.take(indices)
    print(f"   pa.array([1,2,3]).take(np.array([], dtype=float)): {result}")
except Exception as e:
    print(f"   ERROR with numpy float indices: {e}")

print("\n3. Testing what PyArrow expects for indices:")
print(f"   PyArrow accepts indices as: PyArrow array or NumPy array with integer dtype")
print(f"   When NumPy creates np.asanyarray([]), it defaults to float64")
print(f"   This is why ArrowExtensionArray.take([]) fails - PyArrow can't take float indices")