import numpy as np
import io
from scipy.io.matlab import savemat, loadmat

print("Testing how non-empty arrays are handled:")
arr_3 = np.array([1, 2, 3])

f_row = io.BytesIO()
savemat(f_row, {'x': arr_3}, oned_as='row')
f_row.seek(0)
loaded = loadmat(f_row)
print(f"[1,2,3] with oned_as='row': {loaded['x'].shape}")

f_col = io.BytesIO()
savemat(f_col, {'x': arr_3}, oned_as='column')
f_col.seek(0)
loaded = loadmat(f_col)
print(f"[1,2,3] with oned_as='column': {loaded['x'].shape}")

print("\nTesting scalars:")
scalar = np.array(5)
f_scalar = io.BytesIO()
savemat(f_scalar, {'x': scalar}, oned_as='row')
f_scalar.seek(0)
loaded = loadmat(f_scalar)
print(f"Scalar 5 with oned_as='row': {loaded['x'].shape}")

print("\nTesting 2D arrays:")
row_vec = np.array([[1, 2, 3]])
col_vec = np.array([[1], [2], [3]])

f_2d_row = io.BytesIO()
savemat(f_2d_row, {'x': row_vec}, oned_as='row')
f_2d_row.seek(0)
loaded = loadmat(f_2d_row)
print(f"2D [[1,2,3]] (already row): {loaded['x'].shape}")

f_2d_col = io.BytesIO()
savemat(f_2d_col, {'x': col_vec}, oned_as='column')
f_2d_col.seek(0)
loaded = loadmat(f_2d_col)
print(f"2D [[1],[2],[3]] (already column): {loaded['x'].shape}")

print("\nTesting empty 2D arrays:")
empty_row = np.zeros((1, 0))
empty_col = np.zeros((0, 1))
empty_2d = np.zeros((0, 0))

f_empty_row = io.BytesIO()
savemat(f_empty_row, {'x': empty_row})
f_empty_row.seek(0)
loaded = loadmat(f_empty_row)
print(f"Empty 2D (1,0): {loaded['x'].shape}")

f_empty_col = io.BytesIO()
savemat(f_empty_col, {'x': empty_col})
f_empty_col.seek(0)
loaded = loadmat(f_empty_col)
print(f"Empty 2D (0,1): {loaded['x'].shape}")

f_empty_2d = io.BytesIO()
savemat(f_empty_2d, {'x': empty_2d})
f_empty_2d.seek(0)
loaded = loadmat(f_empty_2d)
print(f"Empty 2D (0,0): {loaded['x'].shape}")