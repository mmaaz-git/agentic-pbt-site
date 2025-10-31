from hypothesis import given, strategies as st
from Cython.Build.Dependencies import parse_list


@given(
    st.lists(st.text(alphabet=st.characters(
        blacklist_categories=('Cs',),
        blacklist_characters=' ,[]"\'#\t\n',
        max_codepoint=1000),
        min_size=1, max_size=10),
        min_size=1, max_size=5)
)
def test_parse_list_ignores_comments(items):
    items_str = ' '.join(items)
    test_input = items_str + ' # this is a comment'
    result = parse_list(test_input)

    assert result == items, \
        f"Comments should be filtered out: expected {items}, got {result}"

if __name__ == "__main__":
    # Run the test
    test_parse_list_ignores_comments()