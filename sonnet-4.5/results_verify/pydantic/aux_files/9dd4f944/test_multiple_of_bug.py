#!/usr/bin/env python3
"""Test to reproduce the pydantic.v1 multiple_of bug"""

from hypothesis import given, strategies as st, settings
from pydantic.v1 import BaseModel, Field, ValidationError
import pytest

class MultipleOf(BaseModel):
    value: int = Field(multiple_of=5)

# Test with the specific failing input mentioned
def test_specific_failing_input():
    """Test with the specific value mentioned in the bug report"""
    value = 17608513714555794
    print(f"Testing with value: {value}")
    print(f"Value % 5 = {value % 5}")

    try:
        model = MultipleOf(value=value)
        print(f"Model accepted value: {model.value}")
        print(f"This should have raised ValidationError!")
        return False
    except ValidationError as e:
        print(f"ValidationError raised as expected: {e}")
        return True

# Test with the 10^16 + 1 example
def test_large_integer_example():
    """Test with 10^16 + 1 as mentioned in the bug report"""
    value = 10**16 + 1
    print(f"\nTesting with value: {value}")
    print(f"Value % 5 = {value % 5}")

    try:
        model = MultipleOf(value=value)
        print(f"Model accepted value: {model.value}")
        print(f"Model.value % 5 = {model.value % 5}")
        assert model.value % 5 == 0, f"Value {model.value} is not a multiple of 5!"
        return False
    except ValidationError as e:
        print(f"ValidationError raised: {e}")
        return True
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print(f"Bug confirmed: pydantic accepted {value} which is not a multiple of 5")
        return False

# Property-based test
@given(st.integers().filter(lambda x: x % 5 != 0))
@settings(max_examples=100)
def test_multiple_of_rejects_invalid(value):
    """Property-based test that should reject all non-multiples of 5"""
    with pytest.raises(ValidationError):
        MultipleOf(value=value)

if __name__ == "__main__":
    print("=== Testing specific failing input ===")
    test_specific_failing_input()

    print("\n=== Testing 10^16 + 1 example ===")
    test_large_integer_example()

    print("\n=== Running property-based test ===")
    try:
        test_multiple_of_rejects_invalid()
        print("Property-based test passed (no issues found)")
    except Exception as e:
        print(f"Property-based test failed: {e}")
        # Try to find a failing example
        import traceback
        traceback.print_exc()