import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st, settings


null_char_texts = st.text(alphabet=st.sampled_from('abc\x00'), min_size=0, max_size=10)


@given(st.lists(null_char_texts, min_size=1, max_size=10))
@settings(max_examples=500)
def test_str_len_with_null_chars(string_list):
    arr = np.array(string_list)
    result = ns.str_len(arr)
    expected = np.array([len(s) for s in string_list])
    assert np.array_equal(result, expected), "str_len should count null characters"

if __name__ == "__main__":
    test_str_len_with_null_chars()