#!/usr/bin/env python3
"""
Property-based test using Hypothesis that discovers the bug in parse_list.
This test verifies that parse_list correctly parses space-separated lists
by checking that the number of items returned matches the number of input items.
"""

import sys
# Add the Cython environment to the path
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages")

from hypothesis import given, strategies as st, assume, settings
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1)))
@settings(max_examples=1000)
def test_parse_list_space_separated_count(items):
    assume(all(item.strip() for item in items))
    assume(all(' ' not in item and ',' not in item and '"' not in item and "'" not in item for item in items))

    list_str = ' '.join(items)
    result = parse_list(list_str)
    assert len(result) == len(items), f"Expected {len(items)} items, got {len(result)} items. Input: {items!r}, Output: {result!r}"

# Run the test
if __name__ == "__main__":
    try:
        test_parse_list_space_separated_count()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()