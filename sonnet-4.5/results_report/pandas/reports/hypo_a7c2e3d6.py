import pandas as pd
from hypothesis import given, strategies as st, settings, assume


@given(
    st.lists(st.floats(min_value=-1e-300, max_value=1e-300, allow_nan=False, allow_infinity=False), min_size=2, max_size=10),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=100)
def test_cut_handles_tiny_floats_without_crash(values, n_bins):
    assume(len(set(values)) > 1)

    x = pd.Series(values)
    try:
        result = pd.cut(x, bins=n_bins)
    except ValueError as e:
        if "missing values must be missing" in str(e):
            raise AssertionError(f"cut() crashed on valid input: {values}") from e
        raise

if __name__ == "__main__":
    test_cut_handles_tiny_floats_without_crash()