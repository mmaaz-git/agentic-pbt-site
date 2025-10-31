import numpy as np
from pandas.arrays import SparseArray

sparse = SparseArray([0, 0, 1, 2], fill_value=0)
print(f"Original type: {type(sparse)}")
print(f"Original dtype: {sparse.dtype}")

result = sparse.astype(np.float64)
print(f"Result type: {type(result)}")
print(f"Is SparseArray: {isinstance(result, SparseArray)}")

expected_behavior = "SparseArray"
actual_behavior = type(result).__name__
print(f"\nExpected (per docstring): {expected_behavior}")
print(f"Actual: {actual_behavior}")