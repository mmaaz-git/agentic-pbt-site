from hypothesis import given, strategies as st
from Cython.Build.Dependencies import parse_list

# First test the exact example from the bug report
result = parse_list("a b # comment")
print(f"Result for 'a b # comment': {result}")
assert result == ['a', 'b'], f"Expected ['a', 'b'], got {result}"

# Test the hypothesis-based test
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

# Run a specific test
test_parse_list_ignores_comments(['a', 'b'])
print("Test passed!")