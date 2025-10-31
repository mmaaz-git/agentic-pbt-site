import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume


@given(
    series=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=5, max_size=50
    ),
    window=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_rolling_var_non_negative(series, window):
    assume(len(series) >= window)

    s = pd.Series(series)
    rolling_var = s.rolling(window=window).var()

    valid_mask = ~rolling_var.isna()
    if valid_mask.sum() > 0:
        assert (rolling_var[valid_mask] >= -1e-10).all(), \
            f"Variance should be non-negative, got {rolling_var[valid_mask].min()}"

# Run the test
if __name__ == "__main__":
    try:
        test_rolling_var_non_negative()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        # Re-run with the specific failing case to show it
        print("\nDemonstrating with the minimal failing case:")
        series = [0.0, 494699.5, 0.0, 0.0, 0.00390625]
        window = 3
        s = pd.Series(series)
        rolling_var = s.rolling(window=window).var()
        print(f"Series: {series}")
        print(f"Window: {window}")
        print(f"Rolling variance values: {rolling_var.tolist()}")
        print(f"Minimum variance: {rolling_var.min()}")
        print(f"Is minimum variance negative? {rolling_var.min() < 0}")