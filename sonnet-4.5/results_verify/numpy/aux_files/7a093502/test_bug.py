import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=20),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=1000)
def test_slice_none_end_with_step(string_list, start, step):
    arr = np.array(string_list)
    result = ns.slice(arr, start, None, step)

    for i, s in enumerate(arr):
        expected = s[start:None:step]
        actual = result[i]
        assert actual == expected, f"Failed for string='{s}', start={start}, step={step}. Expected '{expected}', got '{actual}'"

if __name__ == "__main__":
    test_slice_none_end_with_step()
    print("Test passed!")