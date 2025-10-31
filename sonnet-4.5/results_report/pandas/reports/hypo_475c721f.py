from hypothesis import given, strategies as st, settings
import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

@given(
    constant_value=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    length=st.integers(min_value=10, max_value=50)
)
@settings(max_examples=50)
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
if __name__ == "__main__":
    print("Running Hypothesis property-based test for autocorrelation_plot with constant series")
    print("=" * 80)

    try:
        test_any_constant_series()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis confirms the bug: autocorrelation_plot raises RuntimeWarning for constant series")