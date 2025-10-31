from hypothesis import given, strategies as st
import pandas.util

@given(st.text(min_size=1))
def test_capitalize_first_letter_length_preservation(s):
    result = pandas.util.capitalize_first_letter(s)
    assert len(result) == len(s)

if __name__ == "__main__":
    test_capitalize_first_letter_length_preservation()