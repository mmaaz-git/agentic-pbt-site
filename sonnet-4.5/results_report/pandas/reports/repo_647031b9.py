from pandas.arrays import SparseArray
import numpy as np

# Test case that demonstrates the bug
sparse = SparseArray([0])
print("Testing SparseArray([0]).argmin():")
try:
    result = sparse.argmin()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting SparseArray([0]).argmax():")
try:
    result = sparse.argmax()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Compare with numpy behavior
print("\nCompare with numpy behavior:")
numpy_array = np.array([0])
print(f"np.argmin([0]): {np.argmin(numpy_array)}")
print(f"np.argmax([0]): {np.argmax(numpy_array)}")

# Test with multiple values all equal to fill value
print("\nTesting SparseArray([0, 0, 0]):")
sparse2 = SparseArray([0, 0, 0])
try:
    result = sparse2.argmin()
    print(f"SparseArray([0, 0, 0]).argmin(): {result}")
except Exception as e:
    print(f"SparseArray([0, 0, 0]).argmin() Error: {type(e).__name__}: {e}")

try:
    result = sparse2.argmax()
    print(f"SparseArray([0, 0, 0]).argmax(): {result}")
except Exception as e:
    print(f"SparseArray([0, 0, 0]).argmax() Error: {type(e).__name__}: {e}")

# Compare with expected behavior from dense array
print("\nExpected behavior (from dense array):")
dense = sparse.to_dense()
print(f"SparseArray([0]).to_dense().argmin(): {np.argmin(dense)}")
print(f"SparseArray([0]).to_dense().argmax(): {np.argmax(dense)}")