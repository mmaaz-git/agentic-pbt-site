from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
import scipy.stats as stats

@given(
    npst.arrays(
        dtype=float,
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=5, max_side=100),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    ),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=300)
def test_percentileofscore_bounds(arr, score):
    percentile = stats.percentileofscore(arr, score)
    assert 0 <= percentile <= 100, \
        f"Percentile should be in [0, 100], got {percentile}"

if __name__ == "__main__":
    test_percentileofscore_bounds()