import io
import numpy as np
from scipy.io.matlab import loadmat, savemat

arr = np.array([])

f_row = io.BytesIO()
savemat(f_row, {'x': arr}, oned_as='row')
f_row.seek(0)
loaded_row = loadmat(f_row)

print(f"Empty array with oned_as='row': {loaded_row['x'].shape}")
print(f"Expected: (1, 0), Actual: {loaded_row['x'].shape}")

f_col = io.BytesIO()
savemat(f_col, {'x': arr}, oned_as='column')
f_col.seek(0)
loaded_col = loadmat(f_col)

print(f"Empty array with oned_as='column': {loaded_col['x'].shape}")
print(f"Expected: (0, 1), Actual: {loaded_col['x'].shape}")

non_empty = np.array([1, 2, 3])
f_row_ne = io.BytesIO()
savemat(f_row_ne, {'x': non_empty}, oned_as='row')
f_row_ne.seek(0)
loaded_row_ne = loadmat(f_row_ne)
print(f"\nNon-empty array with oned_as='row': {loaded_row_ne['x'].shape}")