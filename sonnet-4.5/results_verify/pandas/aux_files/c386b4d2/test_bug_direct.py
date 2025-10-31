import pandas.errors
import pytest


class DummyClass:
    pass


def test_with_invalid_methodtype(invalid_methodtype):
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    with pytest.raises(ValueError) as exc_info:
        pandas.errors.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)

    error_message = str(exc_info.value)
    print(f"Error message: {error_message}")

    for valid_type in valid_types:
        assert valid_type in error_message, f"Valid type '{valid_type}' not found in error message"

    assert invalid_methodtype in error_message, f"Invalid input '{invalid_methodtype}' not found in error message"

    msg_start = error_message.split(',')[0]
    assert invalid_methodtype not in msg_start, f"Invalid input '{invalid_methodtype}' should not be in first part of message"

    print("All assertions passed!")


# Test with the failing input
test_with_invalid_methodtype("0")