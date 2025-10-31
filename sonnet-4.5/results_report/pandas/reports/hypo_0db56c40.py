import pandas as pd
from hypothesis import given, strategies as st, settings, assume


@given(
    st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False), min_size=10, max_size=100),
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=1, max_value=5)
)
@settings(max_examples=500)
def test_cut_bins_match_intervals(values, n_bins, precision):
    assume(len(set(values)) > 1)

    x = pd.Series(values)
    result, bins = pd.cut(x, bins=n_bins, retbins=True, precision=precision)

    categories = result.cat.categories
    for i, interval in enumerate(categories):
        assert interval.left == bins[i], \
            f"Interval {i} left boundary ({interval.left}) doesn't match bins[{i}] ({bins[i]})"
        assert interval.right == bins[i+1], \
            f"Interval {i} right boundary ({interval.right}) doesn't match bins[{i+1}] ({bins[i+1]})"

if __name__ == "__main__":
    test_cut_bins_match_intervals()