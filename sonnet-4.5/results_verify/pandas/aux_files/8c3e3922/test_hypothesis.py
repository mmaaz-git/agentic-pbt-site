from hypothesis import given, strategies as st
import pandas.errors as errors
import pytest


class DummyClass:
    pass


@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_validation_message_format(invalid_methodtype):
    with pytest.raises(ValueError) as exc_info:
        errors.AbstractMethodError(DummyClass, methodtype=invalid_methodtype)

    error_msg = str(exc_info.value)

    valid_types = {"method", "classmethod", "staticmethod", "property"}

    assert f"methodtype must be one of {invalid_methodtype}" not in error_msg, \
        f"Bug: Error message incorrectly says 'methodtype must be one of {invalid_methodtype}'"

if __name__ == "__main__":
    # Run with a specific failing case
    test_abstract_method_error_validation_message_format("")
    print("Hypothesis test passed with empty string")
    test_abstract_method_error_validation_message_format("invalid_type")
    print("Hypothesis test passed with 'invalid_type'")