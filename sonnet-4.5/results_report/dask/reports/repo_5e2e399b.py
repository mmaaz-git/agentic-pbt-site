import pandas as pd
from dask.dataframe.utils import _maybe_sort

df = pd.DataFrame(
    {'A': [2, 1], 'B': [4, 3]},
    index=pd.Index([10, 20], name='A')
)

print(f"Before: df.index.names = {df.index.names}")

result = _maybe_sort(df, check_index=True)

print(f"After: result.index.names = {result.index.names}")