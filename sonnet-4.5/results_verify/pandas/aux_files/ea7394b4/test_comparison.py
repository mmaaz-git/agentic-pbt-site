import numpy as np
import pandas as pd
import pyarrow as pa

print("Testing different array types with empty indices")
print("=" * 60)

# Test 1: NumPy array
print("\n1. NumPy array with empty indices:")
print("-" * 40)
try:
    np_array = np.array([1, 2, 3])
    result = np.take(np_array, [])
    print(f"np.take(np.array([1, 2, 3]), []): {result}")
    print(f"Result type: {type(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Pandas standard array
print("\n2. Pandas standard array with empty indices:")
print("-" * 40)
try:
    pd_array = pd.array([1, 2, 3])
    result = pd_array.take([])
    print(f"pd.array([1, 2, 3]).take([]): {result}")
    print(f"Result type: {type(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Pandas nullable int array
print("\n3. Pandas nullable int array with empty indices:")
print("-" * 40)
try:
    pd_nullable_array = pd.array([1, 2, 3], dtype='Int64')
    result = pd_nullable_array.take([])
    print(f"pd.array([1, 2, 3], dtype='Int64').take([]): {result}")
    print(f"Result type: {type(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Pandas string array
print("\n4. Pandas string array with empty indices:")
print("-" * 40)
try:
    pd_str_array = pd.array(['a', 'b', 'c'], dtype='string')
    result = pd_str_array.take([])
    print(f"pd.array(['a', 'b', 'c'], dtype='string').take([]): {result}")
    print(f"Result type: {type(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"Error: {e}")

# Test 5: Check what numpy.asanyarray does with empty list
print("\n5. numpy.asanyarray behavior with empty list:")
print("-" * 40)
empty_indices = []
numpy_converted = np.asanyarray(empty_indices)
print(f"np.asanyarray([]): {numpy_converted}")
print(f"dtype: {numpy_converted.dtype}")
print(f"shape: {numpy_converted.shape}")

# Test 6: What dtype should we use?
print("\n6. numpy.asanyarray with explicit dtype:")
print("-" * 40)
int_indices = np.asanyarray([], dtype=np.intp)
print(f"np.asanyarray([], dtype=np.intp): {int_indices}")
print(f"dtype: {int_indices.dtype}")

# Test 7: Test pyarrow directly
print("\n7. PyArrow direct take with empty indices:")
print("-" * 40)
try:
    pa_array = pa.array([1, 2, 3], type=pa.int64())
    int_indices_array = np.asanyarray([], dtype=np.intp)
    result = pa_array.take(int_indices_array)
    print(f"PyArrow array.take(empty int indices): {result}")
except Exception as e:
    print(f"Error with int indices: {e}")

try:
    pa_array = pa.array([1, 2, 3], type=pa.int64())
    float_indices_array = np.asanyarray([])  # This will be float64
    result = pa_array.take(float_indices_array)
    print(f"PyArrow array.take(empty float indices): {result}")
except Exception as e:
    print(f"Error with float indices: {e}")

# Test 8: Pandas Series behavior
print("\n8. Pandas Series with ArrowExtensionArray:")
print("-" * 40)
try:
    series = pd.Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int64()))
    result = series.iloc[[]]  # This uses take internally
    print(f"Series.iloc[[]] works: {result}")
    print(f"dtype: {result.dtype}")
except Exception as e:
    print(f"Error: {e}")