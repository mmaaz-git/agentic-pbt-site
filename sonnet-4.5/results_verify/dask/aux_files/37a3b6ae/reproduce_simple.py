import pandas as pd
import dask.dataframe as dd

print("Testing simple case from bug report:")
df = pd.DataFrame({'a': [1, 2]})
df.index = pd.Index([10, 11], name='idx')

print("\nOriginal pandas DataFrame:")
print(df)
print(f"Index: {df.index.tolist()}")

ddf = dd.from_pandas(df, npartitions=2)
result = ddf.reset_index().compute()

print("\nDask DataFrame after reset_index():")
print(result)
print(f"Result index: {result.index.tolist()}")

# What pandas would produce
expected = df.reset_index()
print("\nExpected (pandas) DataFrame after reset_index():")
print(expected)
print(f"Expected index: {expected.index.tolist()}")

print("\nAre they equal?", result.equals(expected))