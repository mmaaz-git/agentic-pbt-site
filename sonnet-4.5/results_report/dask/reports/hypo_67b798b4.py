import pandas as pd
import numpy as np
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings

@given(
    st.integers(min_value=10, max_value=100),
    st.sampled_from(['1H', '2H', '6H', '1D', '2D', '1W']),
)
@settings(max_examples=200)
def test_resample_matches_pandas(n_points, rule):
    """
    Metamorphic property: Dask resample results should match pandas resample.
    """
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1H')
    np.random.seed(42)
    data = np.random.randn(n_points)

    pandas_series = pd.Series(data, index=dates)
    dask_series = dd.from_pandas(pandas_series, npartitions=4)

    for method in ['sum', 'mean', 'min', 'max', 'count']:
        pandas_result = getattr(pandas_series.resample(rule), method)()
        dask_result = getattr(dask_series.resample(rule), method)().compute()

        pd.testing.assert_series_equal(pandas_result, dask_result, check_dtype=False, rtol=1e-10)

if __name__ == "__main__":
    test_resample_matches_pandas()