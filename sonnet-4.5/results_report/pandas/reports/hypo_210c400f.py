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

if __name__ == "__main__":
    # Run the test to find failing examples
    test_lag_plot_negative_lag()