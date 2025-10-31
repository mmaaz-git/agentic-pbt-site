import numpy as np
from pandas.core.arrays.sparse import SparseArray

print("Testing SparseArray argmin/argmax crash")
print("=" * 50)

# Test case from bug report
arr = np.array([0, 0])
sparse = SparseArray(arr)

print(f"Array: {arr}")
print(f"Fill value: {sparse.fill_value}")
print(f"Sparse values: {sparse.sp_values}")
print()

print(f"np.argmin(arr): {np.argmin(arr)}")
print(f"np.argmax(arr): {np.argmax(arr)}")
print()

try:
    result = sparse.argmin()
    print(f"sparse.argmin(): {result}")
except Exception as e:
    print(f"sparse.argmin() raised: {type(e).__name__}: {e}")

try:
    result = sparse.argmax()
    print(f"sparse.argmax(): {result}")
except Exception as e:
    print(f"sparse.argmax() raised: {type(e).__name__}: {e}")