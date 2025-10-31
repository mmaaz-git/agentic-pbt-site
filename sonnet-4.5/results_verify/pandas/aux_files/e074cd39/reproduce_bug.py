import numpy as np
import pandas as pd
from pandas.api.extensions import take

index = pd.Index([10.0, 20.0, 30.0])
arr = np.array([10.0, 20.0, 30.0])

index_result = take(index, [0, -1, 2], allow_fill=True, fill_value=99.0)
array_result = take(arr, [0, -1, 2], allow_fill=True, fill_value=99.0)

print("Index result:", list(index_result))
print("Array result:", list(array_result))

assert array_result[1] == 99.0
assert pd.isna(index_result[1])
print(f"\nBug confirmed: Index returns NaN instead of fill_value 99.0")