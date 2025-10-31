import pandas as pd
from hypothesis import given, strategies as st, settings, Verbosity, example

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6), min_size=1, max_size=100))
@example([1.023075029544998, 524288.3368640885, 0.0])
@settings(verbosity=Verbosity.verbose, max_examples=100)
def test_expanding_sum_monotonic_for_nonnegative(data):
    s = pd.Series(data)
    result = s.expanding().sum()

    for i in range(1, len(result)):
        if pd.notna(result.iloc[i]) and pd.notna(result.iloc[i-1]):
            assert result.iloc[i] >= result.iloc[i-1], f"Monotonicity violated at position {i}: {result.iloc[i]} < {result.iloc[i-1]}"

# Run the test
if __name__ == "__main__":
    print("Running property-based test to find failures...")
    test_expanding_sum_monotonic_for_nonnegative()