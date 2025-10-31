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

    try:
        dask_result = ddf.rolling(window=window_value, center=center).mean()
        dask_computed = dask_result.compute()

        pd.testing.assert_frame_equal(
            dask_computed.sort_index(),
            pandas_result.sort_index(),
            check_dtype=False
        )
        print(f"✓ PASS: window={window_value}, center={center}, npartitions={npartitions}")
    except TypeError as e:
        if "unsupported operand type(s) for //: 'str' and 'int'" in str(e):
            print(f"✗ FAIL (Expected): window={window_value}, center={center}, npartitions={npartitions} - {e}")
            raise
        else:
            print(f"✗ FAIL (Unexpected): window={window_value}, center={center}, npartitions={npartitions} - {e}")
            raise


if __name__ == "__main__":
    # Run the test with the specific failing input
    print("\nTesting specific failing input: window_value='1h', center=True, npartitions=2")
    test_rolling_window_type_compatibility(window_value='1h', center=True, npartitions=2)

    print("\nRunning hypothesis tests...")
    test_rolling_window_type_compatibility()