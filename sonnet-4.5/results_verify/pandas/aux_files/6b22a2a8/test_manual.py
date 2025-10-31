import numpy as np
from pandas.core.sparse.api import SparseArray

sparse = SparseArray([1, 2, 3], dtype=np.int64)
result = sparse.astype(np.float64)

print(f"Type of result: {type(result)}")
print(f"Result is SparseArray: {isinstance(result, SparseArray)}")
print(f"Result content: {result}")