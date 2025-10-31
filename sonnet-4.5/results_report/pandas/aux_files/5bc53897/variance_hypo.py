from hypothesis import given, strategies as st, settings, assume, example
import pandas as pd

@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=3, max_size=20),
    window=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=500)
@example(data=[5897791891.464727, -2692142700.7497644, 0.0, 1.0], window=2)
def test_rolling_var_nonnegative(data, window):
    assume(window <= len(data))
    s = pd.Series(data)
    result = s.rolling(window=window).var()
    valid_results = result.dropna()
    for i, val in enumerate(valid_results):
        assert val >= 0, f"Variance should be non-negative at index {i+window-1}, got {val} for data {data}"

# Run the test
if __name__ == "__main__":
    test_rolling_var_nonnegative()