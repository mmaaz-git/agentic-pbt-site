import pandas.errors
import pytest
from hypothesis import given, strategies as st


@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstractmethoderror_invalid_methodtype_error_message_correct_order(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pandas.errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    valid_types = {"method", "classmethod", "staticmethod", "property"}

    parts = error_message.split("got")
    assert len(parts) == 2, f"Expected 'got' in error message: {error_message}"

    first_part = parts[0]
    second_part = parts[1]

    for valid_type in valid_types:
        if valid_type in first_part:
            break
    else:
        raise AssertionError(
            f"Expected valid types {valid_types} to appear before 'got', but error message is: {error_message}"
        )

    assert invalid_methodtype in second_part, \
        f"Expected invalid value '{invalid_methodtype}' to appear after 'got', but error message is: {error_message}"


if __name__ == "__main__":
    # Run the test
    test_abstractmethoderror_invalid_methodtype_error_message_correct_order()