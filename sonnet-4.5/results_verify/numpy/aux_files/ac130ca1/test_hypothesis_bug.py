from hypothesis import given, strategies as st, assume
import numpy as np
import numpy.strings as nps

@given(st.lists(st.just("%s"), min_size=1, max_size=5),
       st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=5))
def test_mod_string_formatting(format_strings, values):
    """Property: mod with %s should contain the value"""
    assume(len(format_strings) == len(values))

    fmt_arr = np.array(format_strings, dtype='U')
    val_arr = np.array(values, dtype='U')

    result = nps.mod(fmt_arr, val_arr)

    # Each result should contain the corresponding value
    for i in range(len(result)):
        assert values[i] in str(result[i])

if __name__ == "__main__":
    # Test with the specific failing input
    print("Testing with specific failing input...")
    format_strings = ['%s']
    values = ['\x00']

    fmt_arr = np.array(format_strings, dtype='U')
    val_arr = np.array(values, dtype='U')

    result = nps.mod(fmt_arr, val_arr)

    print(f"Format: {format_strings}")
    print(f"Values: {repr(values)}")
    print(f"Result: {repr(result)}")
    print(f"Result[0]: {repr(result[0])}")
    print(f"Expected: values[0] ('{repr(values[0])}') should be in result[0]")
    print(f"Test assertion: '{values[0]}' in str(result[0]) = {values[0] in str(result[0])}")

    # Run the actual test
    try:
        test_mod_string_formatting()
    except AssertionError as e:
        print(f"\nHypothesis test failed with: {e}")