from hypothesis import given, strategies as st
import pandas as pd
import pytest


class DummyClass:
    pass


@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstractmethoderror_invalid_methodtype_error_message(methodtype):
    instance = DummyClass()
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    with pytest.raises(ValueError) as exc_info:
        pd.errors.AbstractMethodError(instance, methodtype=methodtype)

    error_msg = str(exc_info.value)
    # The error message should say "methodtype must be one of {valid_types}"
    # NOT "methodtype must be one of {invalid_value}"
    assert f"methodtype must be one of {valid_types}" in error_msg, f"Error message has variables swapped: {error_msg}"


if __name__ == "__main__":
    test_abstractmethoderror_invalid_methodtype_error_message()