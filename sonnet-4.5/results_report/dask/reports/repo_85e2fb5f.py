import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create test data
dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
data = pd.DataFrame({'value': np.arange(len(dates), dtype=float)}, index=dates)

# Pandas result (expected)
pandas_result = data.resample('Q', closed='right', label='right').sum()
print("Pandas result (expected):")
print(pandas_result)
print()

# Dask result (actual)
ddf = dd.from_pandas(data, npartitions=4)
dask_result = ddf.resample('Q', closed='right', label='right').sum().compute()
print("Dask result (actual):")
print(dask_result)
print()

# Show the difference
print("Difference (Pandas - Dask):")
print(pandas_result - dask_result)