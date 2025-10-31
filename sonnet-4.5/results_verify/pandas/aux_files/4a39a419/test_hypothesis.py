#!/usr/bin/env python3
"""Property-based test for nested_to_record bug"""

from hypothesis import given, strategies as st, settings
from pandas.io.json._normalize import nested_to_record
import traceback

print("Running property-based test for nested_to_record with non-string keys")
print("=" * 70)

@given(st.dictionaries(
    keys=st.integers(),
    values=st.one_of(
        st.text(),
        st.dictionaries(keys=st.integers(), values=st.text(), max_size=3)
    ),
    max_size=5
))
@settings(max_examples=10)
def test_nested_to_record_handles_non_string_keys(d):
    """Test that nested_to_record handles dictionaries with non-string keys"""
    print(f"\nTesting with input: {d}")
    try:
        result = nested_to_record(d)
        print(f"  Success! Result: {result}")
        assert isinstance(result, dict)
    except Exception as e:
        print(f"  Failed with {type(e).__name__}: {e}")
        raise

# Run the test
try:
    test_nested_to_record_handles_non_string_keys()
    print("\nAll property-based tests passed!")
except Exception as e:
    print(f"\nProperty-based test failed!")
    print(f"Error: {e}")
    traceback.print_exc()