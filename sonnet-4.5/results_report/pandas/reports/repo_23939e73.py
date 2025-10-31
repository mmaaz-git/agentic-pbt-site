import numpy as np
from pandas.core.sparse.api import SparseArray

sparse = SparseArray([1, 2, 3], dtype=np.int64)
result = sparse.astype(np.float64)

print(f"Result type: {type(result)}")
print(f"Result value: {result}")
print(f"Is SparseArray: {isinstance(result, SparseArray)}")