from hypothesis import given, strategies as st
import pandas.errors as errors
import pytest

@given(st.text(min_size=1))
def test_abstractmethoderror_invalid_methodtype_message(invalid_methodtype):
    valid_types = {"method", "classmethod", "staticmethod", "property"}
    if invalid_methodtype in valid_types:
        return

    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    for valid_type in valid_types:
        assert valid_type in error_message
    assert invalid_methodtype not in f"must be one of {invalid_methodtype}"

# Run the test
if __name__ == "__main__":
    test_abstractmethoderror_invalid_methodtype_message()