import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@st.composite
def string_arrays(draw):
    str_list = draw(st.lists(
        st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=50),
        min_size=1, max_size=20
    ))
    return np.array(str_list, dtype='U')

@settings(max_examples=200)
@given(string_arrays(), st.integers(min_value=0, max_value=10))
def test_slice_with_none_stop_matches_python(arr, start):
    result = nps.slice(arr, start, None)

    for i in range(len(arr)):
        s = str(arr[i])
        start_val = min(start, len(s))
        expected = s[start_val:]
        actual = str(result[i])
        assert actual == expected, f"Failed for string '{s}' with start={start}. Expected '{expected}', got '{actual}'"

if __name__ == "__main__":
    # Run the test
    test_slice_with_none_stop_matches_python()