from hypothesis import given, strategies as st, assume
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text()))
def test_parse_list_bracket_delimited_round_trip(items):
    assume(all('"' not in item for item in items))
    quoted_items = [f'"{item}"' for item in items]
    input_str = '[' + ', '.join(quoted_items) + ']'
    result = parse_list(input_str)
    assert result == items

# Run the test
test_parse_list_bracket_delimited_round_trip()