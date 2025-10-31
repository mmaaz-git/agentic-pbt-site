from hypothesis import given, strategies as st
import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0, max_size=100))
def test_autocorrelation_plot_handles_empty(data):
    series = pd.Series(data)
    fig, ax = plt.subplots()
    try:
        result = pandas.plotting.autocorrelation_plot(series)
        assert result is not None
    except ValueError as e:
        assert "empty" in str(e).lower() or "length" in str(e).lower()
    finally:
        plt.close(fig)

# Run the test
if __name__ == "__main__":
    test_autocorrelation_plot_handles_empty()
    print("Test completed - checking with empty series...")

    # Test specifically with empty series
    empty_series = pd.Series([])
    fig, ax = plt.subplots()
    try:
        result = pandas.plotting.autocorrelation_plot(empty_series)
        print("No error occurred - function handled empty series")
    except ZeroDivisionError as e:
        print(f"ZeroDivisionError occurred: {e}")
    except ValueError as e:
        print(f"ValueError occurred: {e}")
    except Exception as e:
        print(f"Other error occurred: {type(e).__name__}: {e}")
    finally:
        plt.close(fig)