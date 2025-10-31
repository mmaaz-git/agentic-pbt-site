import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings
from dask.dataframe import from_pandas


@given(
    window_value=st.sampled_from(['1h', '2h', '1D', '30min']),
    center=st.booleans(),
    npartitions=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=50)
def test_rolling_window_type_compatibility(window_value, center, npartitions):
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=20, freq='30min'),
        'value': range(20)
    })
    df = df.set_index('time')

    ddf = from_pandas(df, npartitions=npartitions)

    pandas_result = df.rolling(window=window_value, center=center).mean()

    dask_result = ddf.rolling(window=window_value, center=center).mean()
    dask_computed = dask_result.compute()

    pd.testing.assert_frame_equal(
        dask_computed.sort_index(),
        pandas_result.sort_index(),
        check_dtype=False
    )

if __name__ == "__main__":
    # Run the hypothesis test
    test_rolling_window_type_compatibility()