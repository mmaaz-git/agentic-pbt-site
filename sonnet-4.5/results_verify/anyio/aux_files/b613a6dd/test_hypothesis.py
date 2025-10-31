from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.array_algos.masked_accumulations import cumsum

@given(
    values=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    mask_indices=st.sets(st.integers(min_value=0, max_value=49), max_size=50)
)
@settings(max_examples=100)
def test_cumsum_does_not_mutate_input(values, mask_indices):
    arr = np.array(values[:min(len(values), 50)], dtype=np.int64)
    mask = np.array([i in mask_indices for i in range(len(arr))], dtype=bool)

    original_arr = arr.copy()

    result_values, result_mask = cumsum(arr, mask, skipna=True)

    assert np.array_equal(arr, original_arr), \
        f"cumsum mutated input array! Before: {original_arr}, After: {arr}"

if __name__ == "__main__":
    try:
        test_cumsum_does_not_mutate_input()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")