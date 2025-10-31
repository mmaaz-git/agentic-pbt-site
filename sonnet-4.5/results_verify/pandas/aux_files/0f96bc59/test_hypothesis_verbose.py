from hypothesis import given, strategies as st, example
import pandas.util

@given(st.text(min_size=1))
@example('ß')  # Add the specific failing example
def test_capitalize_first_letter_length_preservation(s):
    result = pandas.util.capitalize_first_letter(s)
    if len(result) != len(s):
        print(f"Failed for input: {repr(s)}")
        print(f"Input length: {len(s)}, Result: {repr(result)}, Result length: {len(result)}")
        assert False, f"Length changed for {repr(s)}: {len(s)} -> {len(result)}"

if __name__ == "__main__":
    test_capitalize_first_letter_length_preservation()