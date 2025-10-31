import pandas as pd
from dask.dataframe import from_pandas

df = pd.DataFrame({
    'time': pd.date_range('2020-01-01', periods=10, freq='1h'),
    'value': range(10)
})
df = df.set_index('time')

ddf = from_pandas(df, npartitions=2)

print("Attempting dask rolling with window='2h', center=True...")
try:
    result = ddf.rolling(window='2h', center=True).mean()
    computed = result.compute()
    print("Success! Result computed.")
    print(computed)
except TypeError as e:
    print(f"TypeError occurred: {e}")
    import traceback
    traceback.print_exc()