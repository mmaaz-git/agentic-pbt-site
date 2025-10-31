import pandas as pd
import numpy as np
import dask.dataframe as dd

dates = pd.date_range('2024-01-01', periods=10, freq='1h')
data = np.random.randn(10)

pandas_series = pd.Series(data, index=dates)
dask_series = dd.from_pandas(pandas_series, npartitions=4)

pandas_result = pandas_series.resample('1D').sum()
dask_result = dask_series.resample('1D').sum().compute()