import matplotlib
matplotlib.use('Agg')

import pandas as pd
from hypothesis import given, strategies as st, settings
import pandas.plotting as plotting


@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=2, max_size=100),
    lag=st.integers(min_value=-10, max_value=0)
)
@settings(max_examples=100)
def test_lag_plot_rejects_invalid_lag(data, lag):
    series = pd.Series(data)

    try:
        result = plotting.lag_plot(series, lag=lag)
        assert False, f"lag_plot should reject lag={lag} but it succeeded"
    except ValueError:
        pass

if __name__ == "__main__":
    test_lag_plot_rejects_invalid_lag()
    print("Test completed")