import numpy as np
from pandas.core.arrays.sparse import SparseArray

# Test case with all values equal to fill_value
data = [0]
sparse = SparseArray(data, fill_value=0)

print("Creating SparseArray with data=[0] and fill_value=0")
print(f"SparseArray: {sparse}")
print(f"sp_values: {sparse.sp_values}")
print(f"fill_value: {sparse.fill_value}")

print("\nCalling argmax()...")
try:
    result = sparse.argmax()
    print(f"argmax() result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nCalling argmin()...")
try:
    result = sparse.argmin()
    print(f"argmin() result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Compare with NumPy behavior
print("\nNumPy behavior for comparison:")
arr = np.array([0])
print(f"np.array([0]).argmax() = {arr.argmax()}")
print(f"np.array([0]).argmin() = {arr.argmin()}")