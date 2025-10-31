import numpy as np
from pandas.core.arrays.sparse import SparseArray

# Test Case 1: data=[1], fill_value=1 (from the initial report)
print("Test Case 1: data=[1], fill_value=1")
data = [1]
fill_value = 1
sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)

print(f"Data: {data}")
print(f"Fill value: {fill_value}")
print(f"Sparse nonzero(): {sparse.nonzero()[0]}")
print(f"Dense nonzero():  {dense.nonzero()[0]}")
print(f"Expected: {dense.nonzero()[0]}")
print(f"Match: {np.array_equal(sparse.nonzero()[0], dense.nonzero()[0])}")
print()

# Test Case 2: data=[2, 2, 0, 2, 5], fill_value=2
print("Test Case 2: data=[2, 2, 0, 2, 5], fill_value=2")
data = [2, 2, 0, 2, 5]
fill_value = 2
sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)

print(f"Data: {data}")
print(f"Fill value: {fill_value}")
print(f"Sparse nonzero(): {sparse.nonzero()[0]}")
print(f"Dense nonzero():  {dense.nonzero()[0]}")
print(f"Expected: {dense.nonzero()[0]}")
print(f"Match: {np.array_equal(sparse.nonzero()[0], dense.nonzero()[0])}")
print()

# Additional test case to demonstrate the issue
print("Test Case 3: data=[0, 1, 2, 0, 3], fill_value=0 (should work correctly)")
data = [0, 1, 2, 0, 3]
fill_value = 0
sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)

print(f"Data: {data}")
print(f"Fill value: {fill_value}")
print(f"Sparse nonzero(): {sparse.nonzero()[0]}")
print(f"Dense nonzero():  {dense.nonzero()[0]}")
print(f"Expected: {dense.nonzero()[0]}")
print(f"Match: {np.array_equal(sparse.nonzero()[0], dense.nonzero()[0])}")

# The following lines will raise an assertion error for the failing cases
print("\n" + "="*60)
print("Running assertion checks (will fail for non-zero fill_value)...")
print("="*60 + "\n")

# This will fail
data = [1]
fill_value = 1
sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)
assert np.array_equal(sparse.nonzero()[0], dense.nonzero()[0]), "Test Case 1 Failed!"