#!/usr/bin/env python3
"""Property-based test from the bug report"""

from hypothesis import given, strategies as st, settings
import pandas.io.json._normalize as normalize


@given(st.text(min_size=2))
@settings(max_examples=100)
def test_convert_to_line_delimits_property(json_str):
    result = normalize.convert_to_line_delimits(json_str)

    if json_str[0] == "[" and json_str[-1] == "]":
        pass
    else:
        assert result == json_str, f"Non-list string should be unchanged: {json_str!r} -> {result!r}"


# Run the property test
print("Running property-based test...")
try:
    test_convert_to_line_delimits_property()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Error: {e}")