from hypothesis import given, strategies as st, assume, settings
import pandas as pd

@settings(max_examples=500)
@given(
    values=st.lists(
        st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    ),
    n_bins=st.integers(min_value=2, max_value=20)
)
def test_cut_respects_bin_boundaries(values, n_bins):
    assume(len(set(values)) >= 2)

    result = pd.cut(values, bins=n_bins)

    for i, (val, cat) in enumerate(zip(values, result)):
        if pd.notna(cat):
            left, right = cat.left, cat.right
            assert left < val <= right, \
                f"Value {val} at index {i} not in its assigned bin {cat}"

if __name__ == "__main__":
    test_cut_respects_bin_boundaries()