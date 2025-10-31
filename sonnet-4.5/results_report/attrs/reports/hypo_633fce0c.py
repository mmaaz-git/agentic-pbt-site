#!/usr/bin/env python3
"""
Property-based test demonstrating the or_ validator exception handling bug.
Property: or_ validator should only catch validation exceptions, not all exceptions.
"""

from hypothesis import given, strategies as st, settings
import attr
from attr import validators
import pytest


@given(st.integers())
@settings(max_examples=1)  # We only need one example to demonstrate the bug
def test_or_validator_exception_handling(value):
    """
    Property: or_ validator should only catch validation exceptions, not all exceptions.
    """

    class BuggyValidator:
        def __call__(self, inst, attr, value):
            raise AttributeError("Oops! This is a bug, not a validation error")

    buggy = BuggyValidator()
    normal = validators.instance_of(str)
    combined = validators.or_(buggy, normal)

    @attr.define
    class TestClass:
        x: int = attr.field(validator=combined)

    with pytest.raises(AttributeError, match="Oops"):
        TestClass(x=value)


if __name__ == "__main__":
    # Run the test to demonstrate the failure
    try:
        test_or_validator_exception_handling()
        print("Test passed - no bug detected")
    except AssertionError as e:
        print(f"Test failed - bug confirmed!")
        print(f"AssertionError: {e}")
    except Exception as e:
        print(f"Test execution error: {type(e).__name__}: {e}")