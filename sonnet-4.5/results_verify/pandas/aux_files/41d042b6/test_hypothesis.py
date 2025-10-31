from hypothesis import given, strategies as st, assume, settings
import pandas as pd
import math


@given(
    x=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=10, max_size=100),
    bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_cut_assigns_all_values_to_bins(x, bins):
    assume(len(set(x)) > 1)
    result = pd.cut(x, bins)
    non_na_input = sum(1 for val in x if not (math.isnan(val) if isinstance(val, float) else False))
    non_na_output = result.notna().sum()
    assert non_na_input == non_na_output

# Test with the specific failing input
if __name__ == "__main__":
    print("Testing with specific failing input...")
    x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2250738585072014e-308, -1.0]
    bins = 2
    test_cut_assigns_all_values_to_bins(x, bins)
    print("Test passed!")