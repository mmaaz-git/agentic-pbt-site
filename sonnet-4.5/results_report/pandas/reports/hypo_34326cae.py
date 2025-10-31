#!/usr/bin/env python3

from hypothesis import given, strategies as st, assume, settings, example
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1)))
@settings(max_examples=1000)
@example(['#', '0'])  # Known failing case
def test_parse_list_space_separated_count(items):
    assume(all(item.strip() for item in items))
    assume(all(' ' not in item and ',' not in item and '"' not in item and "'" not in item for item in items))

    list_str = ' '.join(items)
    result = parse_list(list_str)
    assert len(result) == len(items), f"Expected {len(items)} items from input {items!r} (string: {list_str!r}), got {len(result)} items: {result!r}"

# Run the test
if __name__ == "__main__":
    from hypothesis import reproduce_failure
    import sys
    try:
        test_parse_list_space_separated_count()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)