import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume


@given(
    st.lists(st.floats(min_value=0, max_value=1e-300, allow_nan=False, allow_infinity=False), min_size=2, max_size=10),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=100)
def test_cut_preserves_data_with_tiny_floats(values, n_bins):
    assume(len(set(values)) > 1)
    assume(all(v >= 0 for v in values))

    x = pd.Series(values)
    result = pd.cut(x, bins=n_bins)

    assert result.notna().sum() == x.notna().sum(), \
        f"Data loss: {x.notna().sum()} valid inputs became {result.notna().sum()} valid outputs"


if __name__ == "__main__":
    test_cut_preserves_data_with_tiny_floats()