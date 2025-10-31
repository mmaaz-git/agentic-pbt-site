#!/usr/bin/env python3
"""
Property-based test using Hypothesis that discovered the bug in
pandas.plotting.autocorrelation_plot when given empty Series
"""

from hypothesis import given, strategies as st, example
import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0, max_size=100))
@example([])  # Explicitly test the empty case
def test_autocorrelation_plot_handles_empty(data):
    """
    Test that autocorrelation_plot handles edge cases gracefully,
    especially empty Series which should either work or raise a meaningful error
    """
    series = pd.Series(data)
    fig, ax = plt.subplots()
    try:
        result = pandas.plotting.autocorrelation_plot(series)
        # If it succeeds, the result should be valid
        assert result is not None
        print(f"✓ Success with {len(data)} elements")
    except ValueError as e:
        # If it fails, it should raise a meaningful ValueError
        assert "empty" in str(e).lower() or "length" in str(e).lower()
        print(f"✓ Raised meaningful ValueError for {len(data)} elements: {e}")
    except ZeroDivisionError as e:
        # This should NOT happen - it's the bug we're reporting
        print(f"✗ BUG: ZeroDivisionError with {len(data)} elements: {e}")
        raise AssertionError(f"Function crashed with ZeroDivisionError instead of handling empty series gracefully: {e}")
    except Exception as e:
        print(f"✗ Unexpected error with {len(data)} elements: {type(e).__name__}: {e}")
        raise
    finally:
        plt.close(fig)

if __name__ == "__main__":
    # Run the test
    test_autocorrelation_plot_handles_empty()