import pytest
from hypothesis import given, strategies as st, example
import pandas.errors as pd_errors


@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
@example("invalid_type")  # Add explicit failing example
def test_abstract_method_error_invalid_methodtype(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pd_errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_msg = str(exc_info.value)
    assert "methodtype must be one of" in error_msg
    # The bug is that the error message has swapped variables
    # It says "methodtype must be one of invalid_type, got {valid_types} instead"
    # When it should say "methodtype must be one of {valid_types}, got invalid_type instead"


if __name__ == "__main__":
    # Run the hypothesis test
    test_abstract_method_error_invalid_methodtype()