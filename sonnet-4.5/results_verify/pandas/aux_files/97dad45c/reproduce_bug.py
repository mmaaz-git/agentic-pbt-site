import pandas as pd
from pandas.api.extensions import take

sparse = pd.arrays.SparseArray([0, 1, 2, 3, 4])
print(f"Created SparseArray: {sparse}")
print(f"Type: {type(sparse)}")
print(f"Attempting to call take(sparse, [0, 1, 2], allow_fill=False)...")

try:
    result = take(sparse, [0, 1, 2], allow_fill=False)
    print(f"Success! Result: {result}")
except TypeError as e:
    print(f"TypeError: {e}")