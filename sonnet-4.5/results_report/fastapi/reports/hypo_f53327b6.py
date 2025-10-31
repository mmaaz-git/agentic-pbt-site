from hypothesis import given, strategies as st
import numpy as np
from dask.array.slicing import check_index

@given(st.integers(min_value=1, max_value=100))
def test_check_index_error_message_accuracy(dim_size):
    too_long_array = np.array([True] * (dim_size + 1))

    try:
        check_index(0, too_long_array, dim_size)
        assert False, "Should have raised IndexError"
    except IndexError as e:
        error_msg = str(e)

        if "not long enough" in error_msg and too_long_array.size > dim_size:
            raise AssertionError(
                f"Error message says 'not long enough' but array size "
                f"{too_long_array.size} is greater than dimension {dim_size}"
            )

if __name__ == "__main__":
    test_check_index_error_message_accuracy()