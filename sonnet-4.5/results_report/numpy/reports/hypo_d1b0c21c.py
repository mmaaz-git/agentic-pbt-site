import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, assume, settings

def string_arrays():
    return st.lists(
        st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            min_size=0, max_size=50
        ),
        min_size=1, max_size=20
    ).map(lambda lst: np.array(lst, dtype='U'))

def simple_strings():
    return st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=0, max_size=20
    )

@given(string_arrays(), simple_strings(), simple_strings())
@settings(max_examples=1000)
def test_replace_length_calculation(arr, old_str, new_str):
    assume(old_str != '')

    result = nps.replace(arr, old_str, new_str)

    original_lengths = nps.str_len(arr)
    result_lengths = nps.str_len(result)
    counts = nps.count(arr, old_str)
    expected_lengths = original_lengths + counts * (len(new_str) - len(old_str))

    assert np.array_equal(result_lengths, expected_lengths)

# Run the test
test_replace_length_calculation()