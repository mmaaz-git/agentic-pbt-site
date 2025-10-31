import pandas as pd
from dask.dataframe.tseries.resample import _resample_series

# Test case from the bug report
series = pd.Series(
    [0.0, 0.0, 0.0, 0.0, 0.0],
    index=pd.date_range('2000-01-01', periods=5, freq='h')
)

start = series.index[0]
end = series.index[-1]

print("Original series:")
print(series)
print(f"\nSeries start: {start}")
print(f"Series end: {end}")

# Show what pandas resample does
resampled_pandas = series.resample('ME').sum()
print(f"\nPandas resample('ME').sum() creates index at: {resampled_pandas.index[0]}")
print(f"Pandas result index: {resampled_pandas.index.tolist()}")
print(f"Pandas result values: {resampled_pandas.values}")

# Show what pd.date_range creates
expected_range = pd.date_range(start, end, freq='ME', inclusive='both')
print(f"\npd.date_range(start={start}, end={end}, freq='ME', inclusive='both') creates: {expected_range.tolist()}")

# Now try the dask function
try:
    result = _resample_series(
        series=series,
        start=start,
        end=end,
        reindex_closed=None,
        rule='ME',
        resample_kwargs={},
        how='sum',
        fill_value=0,
        how_args=(),
        how_kwargs={}
    )
    print("\nDask _resample_series succeeded!")
    print(f"Result: {result}")
except ValueError as e:
    print(f"\nDask _resample_series raised ValueError: {e}")