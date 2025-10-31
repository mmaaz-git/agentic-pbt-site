from hypothesis import given, strategies as st
import pandas.errors as errors
import pytest


@given(
    invalid_methodtype=st.text(min_size=1).filter(
        lambda x: x not in {"method", "classmethod", "staticmethod", "property"}
    )
)
def test_abstractmethoderror_correct_error_format(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    parts = error_message.split(",", 1)
    first_part = parts[0] if len(parts) > 0 else ""

    assert invalid_methodtype not in first_part, (
        f"Bug: The invalid methodtype '{invalid_methodtype}' should not be in "
        f"'methodtype must be one of X' part. Got: {error_message}"
    )

# Test with the specific failing input manually
def test_specific_case():
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        errors.AbstractMethodError(instance, methodtype='0')

    error_message = str(exc_info.value)
    parts = error_message.split(",", 1)
    first_part = parts[0] if len(parts) > 0 else ""

    assert '0' not in first_part, (
        f"Bug: The invalid methodtype '0' should not be in "
        f"'methodtype must be one of X' part. Got: {error_message}"
    )