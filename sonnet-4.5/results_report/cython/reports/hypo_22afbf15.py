from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@given(st.lists(st.text(min_size=0, max_size=10), min_size=0, max_size=20))
@settings(max_examples=1000)
def test_parse_list_no_crash_on_empty_strings(items):
    quoted_items = [f'"{item}"' for item in items]
    input_str = '[' + ', '.join(quoted_items) + ']'
    result = parse_list(input_str)
    assert isinstance(result, list)

if __name__ == "__main__":
    test_parse_list_no_crash_on_empty_strings()