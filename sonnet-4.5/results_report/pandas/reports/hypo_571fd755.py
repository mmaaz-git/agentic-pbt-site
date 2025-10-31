from hypothesis import given, strategies as st, settings
from pandas.io.json._normalize import convert_to_line_delimits


@settings(max_examples=500)
@given(st.text(min_size=2))
def test_convert_to_line_delimits_only_processes_json_arrays(s):
    result = convert_to_line_delimits(s)
    is_json_array_format = s[0] == '[' and s[-1] == ']'

    if not is_json_array_format:
        assert result == s, (
            f"Non-JSON-array string should be returned unchanged. "
            f"Input: {s!r}, Output: {result!r}"
        )


if __name__ == "__main__":
    test_convert_to_line_delimits_only_processes_json_arrays()