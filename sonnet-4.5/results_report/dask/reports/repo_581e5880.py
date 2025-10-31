import pandas as pd
from dask.dataframe import from_pandas

# Create a simple DataFrame with datetime index
df = pd.DataFrame({
    'time': pd.date_range('2020-01-01', periods=10, freq='1h'),
    'value': range(10)
})
df = df.set_index('time')

# Convert to Dask DataFrame
ddf = from_pandas(df, npartitions=2)

# This will crash with TypeError
try:
    result = ddf.rolling(window='2h', center=True).mean()
    computed = result.compute()
    print("Success! Result shape:", computed.shape)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()