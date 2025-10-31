import pandas as pd
from dask.dataframe.tseries.resample import _resample_series

# Test 1: Data that spans a full month - should work
print("Test 1: Data spanning full month")
series1 = pd.Series(
    range(31),
    index=pd.date_range('2000-01-01', '2000-01-31', freq='D')
)
start1 = series1.index[0]
end1 = series1.index[-1]
try:
    result1 = _resample_series(
        series=series1,
        start=start1,
        end=end1,
        reindex_closed=None,
        rule='ME',
        resample_kwargs={},
        how='sum',
        fill_value=0,
        how_args=(),
        how_kwargs={}
    )
    print(f"Success! Result: {result1}")
except ValueError as e:
    print(f"Failed: {e}")

# Test 2: Daily frequency (not anchor-based) - should work
print("\nTest 2: Daily frequency (not anchor-based)")
series2 = pd.Series(
    [0.0, 0.0, 0.0, 0.0, 0.0],
    index=pd.date_range('2000-01-01', periods=5, freq='h')
)
start2 = series2.index[0]
end2 = series2.index[-1]
try:
    result2 = _resample_series(
        series=series2,
        start=start2,
        end=end2,
        reindex_closed=None,
        rule='D',
        resample_kwargs={},
        how='sum',
        fill_value=0,
        how_args=(),
        how_kwargs={}
    )
    print(f"Success! Result: {result2}")
except ValueError as e:
    print(f"Failed: {e}")

# Test 3: Quarterly end frequency (another anchor-based)
print("\nTest 3: Quarterly end frequency")
series3 = pd.Series(
    [1.0] * 10,
    index=pd.date_range('2000-01-01', periods=10, freq='D')
)
start3 = series3.index[0]
end3 = series3.index[-1]

print(f"Series spans from {start3} to {end3}")
resampled_pandas3 = series3.resample('QE').sum()
print(f"Pandas resample('QE').sum() creates index: {resampled_pandas3.index.tolist()}")
expected_range3 = pd.date_range(start3, end3, freq='QE', inclusive='both')
print(f"pd.date_range creates: {expected_range3.tolist()}")

try:
    result3 = _resample_series(
        series=series3,
        start=start3,
        end=end3,
        reindex_closed=None,
        rule='QE',
        resample_kwargs={},
        how='sum',
        fill_value=0,
        how_args=(),
        how_kwargs={}
    )
    print(f"Success! Result: {result3}")
except ValueError as e:
    print(f"Failed: {e}")

# Test 4: Year-end frequency
print("\nTest 4: Year-end frequency")
series4 = pd.Series(
    [1.0] * 10,
    index=pd.date_range('2000-01-01', periods=10, freq='D')
)
start4 = series4.index[0]
end4 = series4.index[-1]

print(f"Series spans from {start4} to {end4}")
resampled_pandas4 = series4.resample('YE').sum()
print(f"Pandas resample('YE').sum() creates index: {resampled_pandas4.index.tolist()}")
expected_range4 = pd.date_range(start4, end4, freq='YE', inclusive='both')
print(f"pd.date_range creates: {expected_range4.tolist()}")

try:
    result4 = _resample_series(
        series=series4,
        start=start4,
        end=end4,
        reindex_closed=None,
        rule='YE',
        resample_kwargs={},
        how='sum',
        fill_value=0,
        how_args=(),
        how_kwargs={}
    )
    print(f"Success! Result: {result4}")
except ValueError as e:
    print(f"Failed: {e}")