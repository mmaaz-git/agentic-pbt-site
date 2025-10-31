"""Test to reproduce the or_ validator bug report."""

from attr.validators import or_, instance_of
from hypothesis import given, strategies as st
import attr


# First, let's test the basic reproduction case
def test_basic_reproduction():
    """Test the basic bug reproduction from the bug report."""

    def buggy_validator(inst, attr, value):
        """A validator that intentionally raises a KeyError."""
        data = {}
        return data[value]  # This will always raise KeyError

    validator = or_(buggy_validator, instance_of(int))

    class FakeAttr:
        name = "test"

    # Test with a string value that should trigger KeyError
    try:
        validator(None, FakeAttr(), "missing_key")
        print("No exception raised - KeyError was silently caught")
        exception_caught = False
    except KeyError as e:
        print(f"KeyError propagated as expected: {e}")
        exception_caught = True
    except Exception as e:
        print(f"Different exception raised: {type(e).__name__}: {e}")
        exception_caught = True

    return exception_caught


# Property-based test from the bug report
def buggy_validator(inst, attr, value):
    """A validator that intentionally raises a KeyError."""
    data = {}
    result = data[value]  # This will always raise KeyError


@given(st.text())
def test_or_should_not_hide_programming_errors(value):
    """Property test to verify KeyError is hidden."""
    validator = or_(buggy_validator, instance_of(int))
    attr_obj = attr.Attribute(
        name="test", default=None, validator=None, repr=True,
        cmp=None, eq=True, eq_key=None, order=False,
        order_key=None, hash=None, init=True, kw_only=False,
        type=None, converter=None, metadata={}, alias=None
    )

    try:
        validator(None, attr_obj, value)
        # If we get here, no exception was raised
        return "no_exception"
    except KeyError:
        # KeyError propagated (expected behavior)
        return "keyerror"
    except ValueError:
        # Validation error (this should happen if all validators fail)
        return "valueerror"
    except Exception as e:
        # Some other exception
        return f"other_{type(e).__name__}"


# Test with an integer to see if the second validator works
def test_with_integer():
    """Test that or_ works correctly when second validator passes."""

    def buggy_validator(inst, attr, value):
        data = {}
        return data[value]  # Will raise KeyError

    validator = or_(buggy_validator, instance_of(int))

    class FakeAttr:
        name = "test"

    # Test with an integer - should pass via second validator
    try:
        validator(None, FakeAttr(), 42)
        print("Validation passed with integer value")
        return True
    except Exception as e:
        print(f"Unexpected exception with integer: {type(e).__name__}: {e}")
        return False


# Test with other exception types
def test_other_exceptions():
    """Test that or_ catches other programming errors too."""
    results = []

    # AttributeError test
    def attr_error_validator(inst, attr, value):
        obj = object()
        return obj.nonexistent  # Will raise AttributeError

    validator1 = or_(attr_error_validator, instance_of(int))
    class FakeAttr:
        name = "test"

    try:
        validator1(None, FakeAttr(), "test")
        results.append("AttributeError: caught")
    except AttributeError:
        results.append("AttributeError: propagated")
    except ValueError:
        results.append("AttributeError: became ValueError")

    # IndexError test
    def index_error_validator(inst, attr, value):
        lst = []
        return lst[10]  # Will raise IndexError

    validator2 = or_(index_error_validator, instance_of(int))

    try:
        validator2(None, FakeAttr(), "test")
        results.append("IndexError: caught")
    except IndexError:
        results.append("IndexError: propagated")
    except ValueError:
        results.append("IndexError: became ValueError")

    # TypeError from programming error (not validation)
    def type_error_validator(inst, attr, value):
        return len(42)  # Will raise TypeError

    validator3 = or_(type_error_validator, instance_of(int))

    try:
        validator3(None, FakeAttr(), "test")
        results.append("TypeError: caught")
    except TypeError:
        results.append("TypeError: propagated")
    except ValueError:
        results.append("TypeError: became ValueError")

    return results


if __name__ == "__main__":
    print("=== Testing Basic Reproduction ===")
    exception_caught = test_basic_reproduction()

    print("\n=== Testing with Integer Value ===")
    int_passed = test_with_integer()

    print("\n=== Testing Other Exception Types ===")
    other_results = test_other_exceptions()
    for result in other_results:
        print(f"  {result}")

    print("\n=== Running Property-Based Test Sample ===")
    # Run a few samples of the property test
    test_samples = ["test", "key", "", "42", "None"]
    for sample in test_samples:
        result = test_or_should_not_hide_programming_errors(sample)
        print(f"  Value '{sample}': {result}")