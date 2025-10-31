#!/usr/bin/env python3
from hypothesis import given, strategies as st, assume
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text()))
def test_parse_list_bracket_delimited_round_trip(items):
    assume(all('"' not in item for item in items))
    quoted_items = [f'"{item}"' for item in items]
    input_str = '[' + ', '.join(quoted_items) + ']'
    print(f"Testing with items: {items}")
    print(f"Input string: {input_str}")
    try:
        result = parse_list(input_str)
        print(f"Result: {result}")
        assert result == items
        print("Test passed!")
    except Exception as e:
        print(f"Exception raised: {type(e).__name__}: {e}")
        raise

# Run the test with the specific failing input
print("\n=== Testing with items=[''] ===")
items = ['']
quoted_items = [f'"{item}"' for item in items]
input_str = '[' + ', '.join(quoted_items) + ']'
print(f"Items: {items}")
print(f"Input string: {input_str}")
try:
    result = parse_list(input_str)
    print(f"Result: {result}")
    assert result == items
    print("Test passed!")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")