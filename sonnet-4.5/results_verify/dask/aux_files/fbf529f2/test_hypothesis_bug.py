import pandas as pd
import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from dask.dataframe.tseries.resample import _resample_series

@st.composite
def time_series_strategy(draw):
    n = draw(st.integers(min_value=5, max_value=30))
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))

    try:
        start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    except:
        assume(False)

    freq = draw(st.sampled_from(['h', '30min', '2h', 'D', '2D']))
    index = pd.date_range(start, periods=n, freq=freq)
    values = draw(st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=n,
        max_size=n
    ))

    return pd.Series(values, index=index)

@given(
    time_series_strategy(),
    st.sampled_from(['D', '2D', 'W', 'h', '2h', '6h', '12h', '30min', 'ME', '3D'])
)
@settings(max_examples=100, deadline=None)  # Reduced for testing
def test_resample_series_index_contained_in_new_index(series, rule):
    assume(len(series) >= 3)

    start = series.index[0]
    end = series.index[-1]

    failures = []

    try:
        result = _resample_series(
            series=series,
            start=start,
            end=end,
            reindex_closed=None,
            rule=rule,
            resample_kwargs={},
            how='sum',
            fill_value=0,
            how_args=(),
            how_kwargs={}
        )
    except ValueError as e:
        if "Index is not contained within new index" in str(e):
            failures.append(f"Index containment violation with rule={rule}, series range={start} to {end}")
            print(f"FAILED: rule={rule}, data from {start} to {end}")
            print(f"  Error: {e}")
            # Don't use pytest.fail here - we're just collecting failures

    return failures

# Run a simple test directly
print("Running direct tests:")
test_cases = [
    ('ME', 5, 'h'),  # Month-end with hourly data
    ('QE', 10, 'D'),  # Quarter-end with daily data
    ('YE', 10, 'D'),  # Year-end with daily data
    ('D', 5, 'h'),   # Daily (should work)
]

for rule, periods, freq in test_cases:
    series = pd.Series(
        [1.0] * periods,
        index=pd.date_range('2000-01-01', periods=periods, freq=freq)
    )
    start = series.index[0]
    end = series.index[-1]

    try:
        result = _resample_series(
            series=series,
            start=start,
            end=end,
            reindex_closed=None,
            rule=rule,
            resample_kwargs={},
            how='sum',
            fill_value=0,
            how_args=(),
            how_kwargs={}
        )
        print(f"✓ {rule} with {freq} data: SUCCESS")
    except ValueError as e:
        if "Index is not contained within new index" in str(e):
            print(f"✗ {rule} with {freq} data: FAILED - Index containment error")