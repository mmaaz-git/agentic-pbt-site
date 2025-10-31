import numpy as np
from pandas.core.arrays.sparse import SparseArray

# Create arrays where all values equal the fill_value (default is 0)
arr = np.array([0, 0])
sparse = SparseArray(arr)

print(f"Array: {arr}")
print(f"Fill value: {sparse.fill_value}")
print(f"Sparse values: {sparse.sp_values}")
print(f"Number of sparse points: {sparse.sp_index.npoints}")

# Show that numpy handles this case correctly
print(f"\nnp.argmin(arr): {np.argmin(arr)}")
print(f"np.argmax(arr): {np.argmax(arr)}")

# Try to call argmin on the sparse array (this should crash)
try:
    print(f"\nSparse.argmin(): {sparse.argmin()}")
except Exception as e:
    print(f"\nSparse.argmin() raised {type(e).__name__}: {e}")

# Try to call argmax on the sparse array (this should also crash)
try:
    print(f"\nSparse.argmax(): {sparse.argmax()}")
except Exception as e:
    print(f"Sparse.argmax() raised {type(e).__name__}: {e}")