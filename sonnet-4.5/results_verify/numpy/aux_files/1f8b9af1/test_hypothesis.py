#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
import pandas.io.formats.printing as printing
import traceback

@given(
    thing=st.text(),
    escape_chars=st.lists(
        st.sampled_from(['\t', '\n', '\r', 'a', 'b', 'c']),
        min_size=1,
        max_size=3
    )
)
@settings(max_examples=100)
def test_pprint_thing_escape_chars(thing, escape_chars):
    """Test that escape_chars parameter accepts any list of characters."""
    result = printing.pprint_thing(thing, escape_chars=escape_chars)
    assert isinstance(result, str)

print("Running hypothesis test...")
print("=" * 60)

# Run the test
try:
    test_pprint_thing_escape_chars()
    print("\nTest passed for all inputs (shouldn't happen with the bug)")
except Exception as e:
    print(f"\nTest failed as expected with KeyError")
    traceback.print_exc()