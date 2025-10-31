import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.array_algos.masked_reductions import sum as masked_sum
from pandas._libs import missing as libmissing


@given(
    values=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_masked_sum_all_masked_returns_na(values):
    """Test that masked_sum returns NA when all values are masked."""
    arr = np.array(values, dtype=np.int64)
    mask = np.ones(len(arr), dtype=bool)  # All values are masked

    result = masked_sum(arr, mask, skipna=True)
    assert result is libmissing.NA, f"Expected NA but got {result} for all-masked array with values {values}"


# Run the test
if __name__ == "__main__":
    test_masked_sum_all_masked_returns_na()