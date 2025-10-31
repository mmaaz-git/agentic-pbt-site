import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st, settings, assume


@given(
    st.lists(st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=0, max_size=20), min_size=1, max_size=20),
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=0, max_size=3),
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=0, max_size=3)
)
@settings(max_examples=500)
def test_replace_all_occurrences(string_list, old, new):
    assume(old != "")
    arr = np.array(string_list)
    result = ns.replace(arr, old, new)

    for i, s in enumerate(arr):
        expected = s.replace(old, new)
        actual = result[i]
        assert actual == expected, f"Failed for string '{s}': expected '{expected}', got '{actual}'"


if __name__ == "__main__":
    test_replace_all_occurrences()