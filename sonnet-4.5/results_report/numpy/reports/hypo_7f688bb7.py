from hypothesis import given, strategies as st, assume, settings
import numpy as np
import numpy.strings as nps

@given(st.lists(st.just("%s"), min_size=1, max_size=5),
       st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=5))
@settings(max_examples=100)
def test_mod_string_formatting(format_strings, values):
    """Property: mod with %s should contain the value"""
    assume(len(format_strings) == len(values))

    fmt_arr = np.array(format_strings, dtype='U')
    val_arr = np.array(values, dtype='U')

    result = nps.mod(fmt_arr, val_arr)

    # Each result should contain the corresponding value
    for i in range(len(result)):
        assert values[i] in str(result[i]), f"Value {repr(values[i])} not found in result {repr(str(result[i]))}"

# Run the test
if __name__ == "__main__":
    test_mod_string_formatting()