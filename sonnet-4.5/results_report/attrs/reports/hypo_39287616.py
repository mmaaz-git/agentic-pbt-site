#!/usr/bin/env python3
"""Property-based test demonstrating attrs.converters.to_bool float acceptance bug"""

from hypothesis import given, strategies as st, settings
from attrs import converters

@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_to_bool_rejects_floats(val):
    """Test that to_bool rejects all float values as per documentation."""
    try:
        result = converters.to_bool(val)
        assert False, f"to_bool({val!r}) should raise ValueError but returned {result}"
    except ValueError:
        pass  # This is the expected behavior for all floats

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for attrs.converters.to_bool...")
    print("=" * 60)
    try:
        test_to_bool_rejects_floats()
        print("Test passed! All float values correctly raised ValueError.")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis demonstrates that to_bool accepts some float values")
        print("despite the documentation only listing integers 0 and 1.")