import pytest
from hypothesis import given, strategies as st, assume
import pandas.errors as pe


@given(st.text(min_size=1))
def test_abstractmethoderror_error_message_shows_correct_values(invalid_methodtype):
    """
    Property: When AbstractMethodError raises ValueError for invalid methodtype,
    the error message should correctly display:
    1. The set of valid types in the "must be one of X" part
    2. The invalid value provided in the "got Y instead" part

    This is a basic contract: error messages should accurately describe what
    went wrong by showing valid options and the invalid input.
    """
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    assume(invalid_methodtype not in valid_types)

    class DummyClass:
        pass
    obj = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pe.AbstractMethodError(obj, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    assert "methodtype must be one of" in error_message

    msg_parts = error_message.split("got")
    assert len(msg_parts) == 2

    first_part = msg_parts[0]
    second_part = msg_parts[1]

    for valid_type in valid_types:
        assert valid_type in first_part, \
            f"Valid type '{valid_type}' should appear in first part (before 'got'), but got: {error_message}"

    assert invalid_methodtype in second_part, \
        f"Invalid type '{invalid_methodtype}' should appear in second part (after 'got'), but got: {error_message}"


if __name__ == "__main__":
    # Run the test to demonstrate the failure
    test_abstractmethoderror_error_message_shows_correct_values('0')
    print("Test passed!")