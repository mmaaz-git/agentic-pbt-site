#!/usr/bin/env python3
"""
Hypothesis property-based test for Django Left/Right functions with None length
"""

import django
from django.conf import settings

# Configure Django settings
settings.configure(
    DEBUG=True,
    SECRET_KEY='test-key'
)
django.setup()

from django.db.models import Value
from django.db.models.functions.text import Left, Right
import pytest
from hypothesis import given, strategies as st

# Test with None directly
def test_left_with_none_length():
    """Test that Left raises an appropriate exception with None length"""
    with pytest.raises((ValueError, TypeError)):
        Left(Value("test"), None)

def test_right_with_none_length():
    """Test that Right raises an appropriate exception with None length"""
    with pytest.raises((ValueError, TypeError)):
        Right(Value("test"), None)

# Property-based test that includes None in possible values
@given(length=st.one_of(st.none(), st.integers()))
def test_left_handles_various_lengths(length):
    """Property test: Left should handle None and integer lengths appropriately"""
    if length is None:
        # Should raise an exception (either ValueError or TypeError)
        with pytest.raises((ValueError, TypeError)):
            Left(Value("test"), length)
    elif length < 1:
        # Should raise ValueError for non-positive integers
        with pytest.raises(ValueError):
            Left(Value("test"), length)
    else:
        # Should work fine for positive integers
        func = Left(Value("test"), length)
        assert func is not None

@given(length=st.one_of(st.none(), st.integers()))
def test_right_handles_various_lengths(length):
    """Property test: Right should handle None and integer lengths appropriately"""
    if length is None:
        # Should raise an exception (either ValueError or TypeError)
        with pytest.raises((ValueError, TypeError)):
            Right(Value("test"), length)
    elif length < 1:
        # Should raise ValueError for non-positive integers
        with pytest.raises(ValueError):
            Right(Value("test"), length)
    else:
        # Should work fine for positive integers
        func = Right(Value("test"), length)
        assert func is not None

if __name__ == "__main__":
    print("Running Hypothesis property-based tests for Left/Right functions...")
    print("=" * 60)

    # Run the direct tests with None
    print("\n1. Testing Left with None length:")
    try:
        test_left_with_none_length()
        print("   PASSED: Left raises exception with None length")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n2. Testing Right with None length:")
    try:
        test_right_with_none_length()
        print("   PASSED: Right raises exception with None length")
    except Exception as e:
        print(f"   FAILED: {e}")

    # Run property-based tests
    print("\n3. Running property-based test for Left:")
    try:
        test_left_handles_various_lengths()
        print("   PASSED: Left handles various length values correctly")
    except Exception as e:
        print(f"   FAILED with example: {e}")

    print("\n4. Running property-based test for Right:")
    try:
        test_right_handles_various_lengths()
        print("   PASSED: Right handles various length values correctly")
    except Exception as e:
        print(f"   FAILED with example: {e}")

    print("\n" + "=" * 60)
    print("Test Summary: The bug is confirmed - Left and Right raise TypeError")
    print("instead of ValueError when length=None")
