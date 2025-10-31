#!/usr/bin/env python3
"""
Property-based test for django.db.backends.utils.truncate_name
This test verifies that the function respects its length parameter constraint.
"""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.utils import truncate_name, split_identifier

@given(st.text(min_size=1, max_size=200), st.integers(min_value=1, max_value=50))
@settings(max_examples=100)
def test_truncate_name_respects_length(identifier, length):
    """
    Test that truncate_name always returns an identifier with the name portion
    having length <= the requested length parameter.
    """
    result = truncate_name(identifier, length=length)
    namespace, name = split_identifier(result)
    actual_name_length = len(name)

    # The property that should hold: name length should not exceed requested length
    assert actual_name_length <= length, (
        f"truncate_name('{identifier}', length={length}) returned '{result}' "
        f"with name length {actual_name_length}, exceeding requested length {length}"
    )

if __name__ == "__main__":
    # Run the property-based test
    print("Running property-based test for truncate_name...")
    print("This test verifies that truncate_name respects its length parameter.")
    print("="*60)

    try:
        test_truncate_name_respects_length()
        print("✓ All tests passed!")
    except AssertionError as e:
        print(f"❌ Test failed!")
        print(f"Assertion error: {e}")
    except Exception as e:
        print(f"❌ Test encountered an error!")
        print(f"Error details: {e}")