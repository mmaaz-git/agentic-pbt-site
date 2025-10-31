#!/usr/bin/env python3

# First, test the Hypothesis test
from pandas.io.json._normalize import convert_to_line_delimits
from hypothesis import given, strategies as st, assume, settings

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=100))
@settings(max_examples=100, deadline=None)
def test_convert_to_line_delimits_non_list_unchanged(json_str):
    assume(not (json_str.startswith('[') and json_str.endswith(']')))
    result = convert_to_line_delimits(json_str)
    assert result == json_str

# Run the hypothesis test
print("Running Hypothesis test...")
try:
    test_convert_to_line_delimits_non_list_unchanged()
    print("Hypothesis test passed (no bugs found)")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Hypothesis test error: {e}")

# Now test the specific examples
print("\nTesting specific examples:")
print(f"convert_to_line_delimits('0') = {repr(convert_to_line_delimits('0'))}")
print(f"convert_to_line_delimits('123') = {repr(convert_to_line_delimits('123'))}")
print(f"convert_to_line_delimits('{{}}') = {repr(convert_to_line_delimits('{}'))}")
print(f"convert_to_line_delimits('\"hello\"') = {repr(convert_to_line_delimits('\"hello\"'))}")

# Test cases that should work (JSON arrays)
print("\nTesting valid JSON arrays:")
print(f"convert_to_line_delimits('[1,2,3]') = {repr(convert_to_line_delimits('[1,2,3]'))}")
print(f"convert_to_line_delimits('[]') = {repr(convert_to_line_delimits('[]'))}")