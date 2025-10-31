"""Test the hypothesis test case from bug report"""
from hypothesis import given, strategies as st
import attr
from attr import validators
import pytest


@given(st.integers())
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

    # The test expects AttributeError to be raised
    # Let's see what actually happens
    try:
        TestClass(x=value)
        print(f"No exception for value={value}")
        return False
    except AttributeError:
        print(f"AttributeError raised for value={value} - EXPECTED")
        return True
    except ValueError as e:
        print(f"ValueError raised for value={value}: {e} - BUG!")
        return False
    except Exception as e:
        print(f"Other exception for value={value}: {type(e).__name__}: {e}")
        return False

# Run the test manually (without hypothesis decorator)
def manual_test(value):
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

    # The test expects AttributeError to be raised
    # Let's see what actually happens
    try:
        TestClass(x=value)
        print(f"No exception for value={value}")
        return False
    except AttributeError:
        print(f"AttributeError raised for value={value} - EXPECTED")
        return True
    except ValueError as e:
        print(f"ValueError raised for value={value}: {e} - BUG!")
        return False
    except Exception as e:
        print(f"Other exception for value={value}: {type(e).__name__}: {e}")
        return False

print("Testing with a few example values:")
for val in [0, 42, -1, 100]:
    result = manual_test(val)
    if not result:
        print(f"  -> Test FAILED for value {val}")
    else:
        print(f"  -> Test PASSED for value {val}")