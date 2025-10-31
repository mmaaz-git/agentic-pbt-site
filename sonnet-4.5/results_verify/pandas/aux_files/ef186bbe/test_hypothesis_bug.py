from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@settings(max_examples=100, deadline=None)
@given(
    value=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    length=st.integers(min_value=2, max_value=100)
)
def test_autocorrelation_plot_constant_series(value, length):
    series = pd.Series([value] * length)
    result = pd.plotting.autocorrelation_plot(series)

    lines = result.get_lines()
    if lines:
        ydata = lines[-1].get_ydata()
        assert not np.all(np.isnan(ydata)), "All autocorrelation values are NaN"

    plt.close('all')

if __name__ == "__main__":
    # Run the test
    test_autocorrelation_plot_constant_series()