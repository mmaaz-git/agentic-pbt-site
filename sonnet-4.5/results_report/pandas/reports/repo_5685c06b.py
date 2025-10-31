import pandas as pd
from pandas.api.extensions import take

sparse = pd.arrays.SparseArray([0, 1, 2, 3, 4])
result = take(sparse, [0, 1, 2], allow_fill=False)
print(result)