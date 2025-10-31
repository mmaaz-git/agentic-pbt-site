#!/usr/bin/env python3
"""Test the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings, Verbosity
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import transform
from typing import Annotated
import pytest

@given(st.integers(), st.lists(st.integers(), min_size=1))
@settings(verbosity=Verbosity.verbose, max_examples=50)
def test_not_in_rejects_excluded_values(value, excluded_values):
    """The exact test from the bug report"""
    if value not in excluded_values:
        return  # Skip values not in the exclusion list

    class Model(BaseModel):
        field: Annotated[int, transform(lambda x: x).not_in(excluded_values)]

    with pytest.raises(ValidationError):
        Model(field=value)

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test from bug report...")
    try:
        test_not_in_rejects_excluded_values()
        print("✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test specific example from bug report
    print("\nTesting specific example: value=2, excluded_values=[1, 2, 3]")
    try:
        value = 2
        excluded_values = [1, 2, 3]

        class Model(BaseModel):
            field: Annotated[int, transform(lambda x: x).not_in(excluded_values)]

        try:
            result = Model(field=value)
            print(f"✗ Model incorrectly accepted value {value}")
        except ValidationError:
            print(f"✓ Model correctly rejected value {value}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")