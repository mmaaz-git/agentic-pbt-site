import pandas as pd
import dask.dataframe as dd

df_pandas = pd.DataFrame({'text': ['hello', 'world']})
print(f"Original dtype: {df_pandas['text'].dtype}")

df_dask = dd.from_pandas(df_pandas, npartitions=1)
result = df_dask.compute()
print(f"After round-trip: {result['text'].dtype}")

try:
    assert df_pandas['text'].dtype == result['text'].dtype
    print("Assertion passed - dtypes are the same")
except AssertionError:
    print("AssertionError: Dtype changed from object to string[pyarrow]")
    print(f"  Original: {df_pandas['text'].dtype}")
    print(f"  After:    {result['text'].dtype}")