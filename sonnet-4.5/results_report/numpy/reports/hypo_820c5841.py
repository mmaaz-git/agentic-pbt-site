import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, example

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
@example(['hello'])
def test_slice_with_none_stop(str_list):
    """Property: nps.slice(arr, start, None) should behave like Python arr[start:]"""
    arr = np.array(str_list, dtype='U')
    result = nps.slice(arr, 0, None)

    for i in range(len(arr)):
        expected = str_list[i][0:]
        assert result[i] == expected, f"slice(arr, 0, None) failed: got '{result[i]}', expected '{expected}'"

if __name__ == "__main__":
    test_slice_with_none_stop()