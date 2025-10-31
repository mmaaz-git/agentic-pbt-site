from hypothesis import given, settings, strategies as st, example
import numpy as np
from pandas.api.extensions import take
import pytest

@given(
    arr=st.lists(st.integers(), min_size=5, max_size=100),
    negative_val=st.integers(max_value=-2)
)
@settings(max_examples=50)  # Reduced for faster testing
@example(arr=[0, 0, 0, 0, 0], negative_val=-9_223_372_036_854_775_809)
def test_take_allow_fill_invalid_negative_raises_valueerror(arr, negative_val):
    np_arr = np.array(arr)
    try:
        result = take(np_arr, [0, negative_val], allow_fill=True)
        print(f"No exception for arr={arr[:5]}..., negative_val={negative_val}")
        assert False, "Expected ValueError or OverflowError"
    except ValueError:
        pass  # Expected
    except OverflowError as e:
        print(f"OverflowError for arr={arr[:5]}..., negative_val={negative_val}: {e}")
        # This is the bug - we expect ValueError but get OverflowError
        raise AssertionError(f"Got OverflowError instead of ValueError for negative_val={negative_val}")

if __name__ == "__main__":
    test_take_allow_fill_invalid_negative_raises_valueerror()