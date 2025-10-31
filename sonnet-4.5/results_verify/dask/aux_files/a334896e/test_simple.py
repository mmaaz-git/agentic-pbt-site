import dask.dataframe as dd
import pandas as pd

pdf = pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['hello']})

print("Input dtype:", pdf['c'].dtype)
print("Input DataFrame:")
print(pdf)
print()

ddf = dd.from_pandas(pdf, npartitions=2)
result = ddf.compute()

print("Output dtype:", result['c'].dtype)
print("Output DataFrame:")
print(result)
print()

print("Are dtypes equal?", pdf['c'].dtype == result['c'].dtype)
assert pdf['c'].dtype == result['c'].dtype