import dask
import dask.dataframe as dd
import pandas as pd

# Test with default config
pdf = pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['hello']})

print("=== With default config (dataframe.convert-string=None -> True) ===")
print("Input dtype:", pdf['c'].dtype)

ddf = dd.from_pandas(pdf, npartitions=2)
result = ddf.compute()

print("Output dtype:", result['c'].dtype)
print("Are dtypes equal?", pdf['c'].dtype == result['c'].dtype)
print()

# Now test with config disabled
print("=== With dataframe.convert-string=False ===")
with dask.config.set({"dataframe.convert-string": False}):
    ddf = dd.from_pandas(pdf, npartitions=2)
    result = ddf.compute()

    print("Output dtype:", result['c'].dtype)
    print("Are dtypes equal?", pdf['c'].dtype == result['c'].dtype)