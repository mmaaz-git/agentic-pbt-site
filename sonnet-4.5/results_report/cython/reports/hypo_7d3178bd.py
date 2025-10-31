from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@settings(max_examples=1000)
@given(st.lists(st.text(min_size=0, max_size=20), min_size=0, max_size=10))
def test_parse_list_quoted_bracket_format_no_crash(items):
    s = '[' + ', '.join(f'"{item}"' for item in items) + ']'
    result = parse_list(s)


if __name__ == "__main__":
    # Run the property-based test
    test_parse_list_quoted_bracket_format_no_crash()