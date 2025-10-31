import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from hypothesis import given, strategies as st, settings

@given(
    constant_value=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    length=st.integers(min_value=10, max_value=50)
)
@settings(max_examples=10)
def test_any_constant_series(constant_value, length):
    """
    Property: autocorrelation_plot should handle any constant series without warnings.
    """
    series = pd.Series([constant_value] * length)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        result = pandas.plotting.autocorrelation_plot(series)
        plt.close('all')

        runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, \
            f"autocorrelation_plot should not raise warnings for constant={constant_value}"

# Run the test
print("Running Hypothesis test for constant series...")
try:
    test_any_constant_series()
except AssertionError as e:
    print(f"Test failed as expected: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")