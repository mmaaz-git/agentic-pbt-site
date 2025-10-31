from hypothesis import given, strategies as st, assume
import json
from pandas.io.json._normalize import convert_to_line_delimits

@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers()))
def test_convert_to_line_delimits_preserves_non_lists(d):
    assume(len(d) > 0)

    json_obj = json.dumps(d)
    result = convert_to_line_delimits(json_obj)

    assert json_obj == result, f"Non-list JSON should be unchanged. Input: {repr(json_obj)}, Output: {repr(result)}"

# Run the test
if __name__ == "__main__":
    test_convert_to_line_delimits_preserves_non_lists()
    print("Test passed!")