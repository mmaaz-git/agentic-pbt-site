from hypothesis import given, settings, strategies as st
import pandas as pd
import pandas.plotting
import pytest
import matplotlib
matplotlib.use('Agg')

@settings(max_examples=200)
@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2, max_size=100),
    lag=st.integers(min_value=-10, max_value=-1)
)
def test_lag_plot_negative_lag(data, lag):
    series = pd.Series(data)

    with pytest.raises(ValueError):
        pandas.plotting.lag_plot(series, lag=lag)

# Run the test
if __name__ == "__main__":
    # Test with a simple case first
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        result = pandas.plotting.lag_plot(series, lag=-1)
        print(f"Test FAILED: Function succeeded with lag=-1, returned: {result}")
        print(f"The function should have raised ValueError but it didn't")
    except ValueError as e:
        print(f"Test PASSED: Function correctly raised ValueError: {e}")