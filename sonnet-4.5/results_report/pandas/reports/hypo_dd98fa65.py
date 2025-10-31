from hypothesis import given, strategies as st
from pandas.io.json._normalize import convert_to_line_delimits

@given(st.text())
def test_convert_to_line_delimits_no_crash(s):
    result = convert_to_line_delimits(s)

# Run the test
if __name__ == "__main__":
    test_convert_to_line_delimits_no_crash()