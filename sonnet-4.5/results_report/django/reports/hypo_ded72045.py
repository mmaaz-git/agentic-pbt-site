#!/usr/bin/env python3
"""Property-based test for django.core.management.utils.handle_extensions."""

from hypothesis import given, strategies as st, settings, example
from django.core.management.utils import handle_extensions

@given(st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz,. ", min_size=1, max_size=30)))
@example(['py,,js'])  # The failing example from the initial report
@settings(max_examples=100)
def test_handle_extensions_no_dot_only_extension(extensions):
    """
    Property: handle_extensions should never return '.' as an extension
    A lone dot is not a valid file extension.
    """
    result = handle_extensions(extensions)
    assert '.' not in result, f"Result contains invalid extension '.': {result}"

if __name__ == "__main__":
    print("Running property-based test for handle_extensions...")
    print("Testing that handle_extensions never returns '.' as an extension")
    print()

    try:
        test_handle_extensions_no_dot_only_extension()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis test checks that handle_extensions should never return '.' as a standalone extension.")
        print("A lone dot is not semantically valid as a file extension.")