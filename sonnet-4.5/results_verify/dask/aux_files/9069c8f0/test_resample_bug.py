"""Test to reproduce the dask resample bug"""
import pandas as pd
import numpy as np
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings
import traceback

# First, test the specific failing case mentioned in the bug report
def test_specific_case():
    print("Testing specific case: n_points=10, rule='1D'")
    dates = pd.date_range('2024-01-01', periods=10, freq='1h')
    np.random.seed(42)
    data = np.random.randn(10)

    pandas_series = pd.Series(data, index=dates)
    dask_series = dd.from_pandas(pandas_series, npartitions=4)

    # Test pandas result
    pandas_result = pandas_series.resample('1D').sum()
    print(f"Pandas result:\n{pandas_result}")

    # Test dask result
    try:
        dask_result = dask_series.resample('1D').sum().compute()
        print(f"Dask result:\n{dask_result}")

        # Compare results if both succeed
        pd.testing.assert_series_equal(pandas_result, dask_result, check_dtype=False, rtol=1e-10)
        print("Results match!")
    except Exception as e:
        print(f"Dask failed with error: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

    return True

# Test the property-based test from the bug report
@given(
    st.integers(min_value=10, max_value=100),
    st.sampled_from(['1H', '2H', '6H', '1D', '2D', '1W']),
)
@settings(max_examples=50, deadline=None)
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
    print("=" * 60)
    print("TESTING DASK RESAMPLE BUG")
    print("=" * 60)

    # Test the specific case
    result = test_specific_case()

    if not result:
        print("\n" + "=" * 60)
        print("BUG CONFIRMED: Dask resample fails with AssertionError")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("NO BUG: Dask resample works correctly")
        print("=" * 60)

    # Run property-based testing
    print("\n" + "=" * 60)
    print("RUNNING PROPERTY-BASED TESTS")
    print("=" * 60)

    try:
        test_resample_matches_pandas()
        print("All property-based tests passed!")
    except Exception as e:
        print(f"Property-based tests failed: {e}")
        traceback.print_exc()