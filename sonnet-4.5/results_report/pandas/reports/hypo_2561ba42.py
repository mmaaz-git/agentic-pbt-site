import math

import pandas as pd
from hypothesis import assume, given, settings, strategies as st


@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        min_size=1,
        max_size=100,
    ),
    st.integers(min_value=1, max_value=10),
)
@settings(max_examples=500)
def test_rolling_min_max_bounds(data, window):
    assume(window <= len(data))
    s = pd.Series(data)

    rolling_min = s.rolling(window=window).min()
    rolling_max = s.rolling(window=window).max()
    rolling_mean = s.rolling(window=window).mean()

    for i in range(window - 1, len(data)):
        min_val = rolling_min.iloc[i]
        max_val = rolling_max.iloc[i]
        mean_val = rolling_mean.iloc[i]

        assert min_val <= mean_val or math.isclose(min_val, mean_val, rel_tol=1e-9, abs_tol=1e-9), \
            f"At {i}: rolling_min {min_val} > rolling_mean {mean_val}"
        assert mean_val <= max_val or math.isclose(mean_val, max_val, rel_tol=1e-9, abs_tol=1e-9), \
            f"At {i}: rolling_mean {mean_val} > rolling_max {max_val}"


def test_specific_failing_input():
    data = [1.0, -4294967297.0, 0.99999, 0.0, 0.0, 1.6675355247098508e-21]
    window = 3
    s = pd.Series(data)

    rolling_min = s.rolling(window=window).min()
    rolling_max = s.rolling(window=window).max()
    rolling_mean = s.rolling(window=window).mean()

    for i in range(window - 1, len(data)):
        min_val = rolling_min.iloc[i]
        max_val = rolling_max.iloc[i]
        mean_val = rolling_mean.iloc[i]

        assert min_val <= mean_val or math.isclose(min_val, mean_val, rel_tol=1e-9, abs_tol=1e-9), \
            f"At {i}: rolling_min {min_val} > rolling_mean {mean_val}"
        assert mean_val <= max_val or math.isclose(mean_val, max_val, rel_tol=1e-9, abs_tol=1e-9), \
            f"At {i}: rolling_mean {mean_val} > rolling_max {max_val}"


if __name__ == "__main__":
    print("Running property-based test...")
    print("Testing with the specific failing input:")
    data = [1.0, -4294967297.0, 0.99999, 0.0, 0.0, 1.6675355247098508e-21]
    window = 3
    print(f"data={data}")
    print(f"window={window}")

    try:
        test_specific_failing_input()
        print("TEST PASSED (unexpected!)")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\nRunning full hypothesis test suite...")
    from hypothesis import Verbosity
    try:
        test_rolling_min_max_bounds()
        print("All tests passed")
    except Exception as e:
        print(f"Test failed: {e}")