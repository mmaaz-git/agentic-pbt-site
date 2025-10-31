from hypothesis import given, strategies as st
import pandas.errors
import pytest


class DummyClass:
    pass


@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_invalid_methodtype(invalid_type):
    with pytest.raises(ValueError) as excinfo:
        pandas.errors.AbstractMethodError(DummyClass(), methodtype=invalid_type)

    error_msg = str(excinfo.value)
    assert "methodtype must be one of" in error_msg


if __name__ == "__main__":
    test_abstract_method_error_invalid_methodtype()