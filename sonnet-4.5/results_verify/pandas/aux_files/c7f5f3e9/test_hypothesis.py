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

# Test with the specific failing input
if __name__ == "__main__":
    # Test the specific failing case
    series = [0.0, 494699.5, 0.0, 0.0, 0.00390625]
    window = 3
    print(f"Testing specific case: series={series}, window={window}")
    test_rolling_var_non_negative(series, window)
    print("Specific case passed!")

    # Run hypothesis tests
    print("\nRunning hypothesis tests...")
    test_rolling_var_non_negative()