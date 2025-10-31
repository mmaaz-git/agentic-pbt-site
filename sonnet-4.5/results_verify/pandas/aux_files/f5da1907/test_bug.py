from hypothesis import given, settings, strategies as st
import pandas.util.version as version_module

@given(st.integers(min_value=0, max_value=1))
@settings(max_examples=1000)
def test_parse_letter_version_integer_zero_bug(number):
    result = version_module._parse_letter_version(None, number)

    if number == 0:
        assert result == ("post", 0), f"Bug: integer 0 treated as falsy, returns {result}"
    else:
        assert result == ("post", 1), f"Expected ('post', 1) for input 1, got {result}"

# Run the test
if __name__ == "__main__":
    test_parse_letter_version_integer_zero_bug()