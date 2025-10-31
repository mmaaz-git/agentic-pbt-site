import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray

print("=== Reproducing the bug from the report ===")
print()

# Test the specific failing case from the bug report
data = [0]
sparse = SparseArray(data, fill_value=0)
print(f"Original array: {sparse.to_dense()}")
print(f"Original fill_value: {sparse.fill_value}")

dtype = pd.SparseDtype(np.float64, fill_value=1)
casted = sparse.astype(dtype)
print(f"After astype with new fill_value:")
print(f"Result: {casted.to_dense()}")
print(f"Expected: [0.]")

if np.array_equal(casted.to_dense(), [1.]):
    print("BUG CONFIRMED: Value was replaced with new fill_value")
elif np.array_equal(casted.to_dense(), [0.]):
    print("NO BUG: Value was preserved correctly")

print("\n=== Testing when the bug occurs ===")

# Case 1: All values equal fill_value
data = [5, 5, 5]
sparse = SparseArray(data, fill_value=5)
dtype = pd.SparseDtype(np.float64, fill_value=10)
result = sparse.astype(dtype).to_dense()
expected = np.array(data).astype(np.float64)
print(f"\nCase 1: All values equal fill_value")
print(f"  Data: {data}, fill_value: 5 -> 10")
print(f"  Expected: {expected}, Actual: {result}")
print(f"  Bug occurs: {not np.array_equal(result, expected)}")

# Case 2: Some values equal fill_value
data = [5, 6, 5]
sparse = SparseArray(data, fill_value=5)
dtype = pd.SparseDtype(np.float64, fill_value=10)
result = sparse.astype(dtype).to_dense()
expected = np.array(data).astype(np.float64)
print(f"\nCase 2: Some values equal fill_value")
print(f"  Data: {data}, fill_value: 5 -> 10")
print(f"  Expected: {expected}, Actual: {result}")
print(f"  Bug occurs: {not np.array_equal(result, expected)}")

# Case 3: No values equal fill_value
data = [6, 7, 8]
sparse = SparseArray(data, fill_value=5)
dtype = pd.SparseDtype(np.float64, fill_value=10)
result = sparse.astype(dtype).to_dense()
expected = np.array(data).astype(np.float64)
print(f"\nCase 3: No values equal fill_value")
print(f"  Data: {data}, fill_value: 5 -> 10")
print(f"  Expected: {expected}, Actual: {result}")
print(f"  Bug occurs: {not np.array_equal(result, expected)}")

print("\n=== Testing the documented example ===")
# This example is from the pandas documentation itself
arr = pd.arrays.SparseArray([0, 0, 1, 2])
print(f"Original: {arr.to_dense()}, fill_value={arr.fill_value}")
result = arr.astype(pd.SparseDtype("float64", fill_value=0.0))
print(f"After astype(SparseDtype('float64', fill_value=0.0)):")
print(f"Result: {result.to_dense()}")
print(f"This works correctly because 1 and 2 are not equal to fill_value")

print("\n=== Testing the fundamental invariant ===")
# The invariant: array.astype(dtype).to_dense() == array.to_dense().astype(dtype)
data = [0, 0, 0]
sparse = SparseArray(data, fill_value=0)
dtype = pd.SparseDtype(np.float64, fill_value=1.0)
method1 = sparse.astype(dtype).to_dense()
method2 = sparse.to_dense().astype(np.float64)
print(f"sparse.astype(dtype).to_dense() = {method1}")
print(f"sparse.to_dense().astype(dtype) = {method2}")
if np.array_equal(method1, method2):
    print("CORRECT: Both methods give the same result")
else:
    print("BUG CONFIRMED: The fundamental invariant is violated!")