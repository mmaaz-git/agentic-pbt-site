from hypothesis import given, strategies as st
import pandas.io.json._normalize as normalize


@given(st.text(min_size=2))
def test_convert_to_line_delimits_property(json_str):
    result = normalize.convert_to_line_delimits(json_str)

    if json_str[0] == "[" and json_str[-1] == "]":
        # This is a JSON array, should be converted to line-delimited format
        pass  # We don't validate the exact format here
    else:
        # Non-array strings should be returned unchanged
        assert result == json_str, f"Non-list string should be unchanged: {json_str!r} -> {result!r}"


# Run the test
if __name__ == "__main__":
    test_convert_to_line_delimits_property()