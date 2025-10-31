import pandas as pd
import dask.dataframe as dd
import numpy as np

dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
data = pd.DataFrame({'value': np.arange(len(dates), dtype=float)}, index=dates)

pandas_result = data.resample('Q', closed='right', label='right').sum()
print("Pandas result:")
print(pandas_result)

ddf = dd.from_pandas(data, npartitions=4)
dask_result = ddf.resample('Q', closed='right', label='right').sum().compute()
print("\nDask result:")
print(dask_result)

print("\nExpected last value: 29394.0")
print(f"Actual last value: {dask_result.iloc[-1, 0]}")
print(f"Bug confirmed: {dask_result.iloc[-1, 0] != pandas_result.iloc[-1, 0]}")