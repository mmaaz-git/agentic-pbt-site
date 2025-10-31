import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [1.0, 2.0, 3.0],
    'c': ['x', 'y', 'z']
})

print("Original dtypes:")
print(df.dtypes)
print()

ddf = dd.from_pandas(df, npartitions=2)
result = ddf.compute()

print("After round-trip dtypes:")
print(result.dtypes)
print()

print(f"Column 'c' dtype changed: {df['c'].dtype} -> {result['c'].dtype}")

try:
    assert df['c'].dtype == result['c'].dtype, f"Expected {df['c'].dtype}, got {result['c'].dtype}"
    print("SUCCESS: dtypes match")
except AssertionError as e:
    print(f"FAILED: {e}")