from hypothesis import given, strategies as st, settings
from pandas.io.json._normalize import convert_to_line_delimits

@given(st.text(min_size=2, max_size=100))
@settings(max_examples=1000)
def test_convert_to_line_delimits_only_processes_arrays(s):
    result = convert_to_line_delimits(s)

    if s[0] == '[' and s[-1] == ']':
        pass
    else:
        assert result == s, f"Non-array input should be returned unchanged"

if __name__ == "__main__":
    test_convert_to_line_delimits_only_processes_arrays()