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
    print(f"Testing with invalid_type='{invalid_type}'")
    print(f"Error message: {error_msg}")


# Test with the specific failing input mentioned
def test_specific_case():
    invalid_type = 'foo'
    with pytest.raises(ValueError) as excinfo:
        pandas.errors.AbstractMethodError(DummyClass(), methodtype=invalid_type)

    error_msg = str(excinfo.value)
    print(f"For invalid_type='foo': {error_msg}")
    assert "methodtype must be one of" in error_msg


if __name__ == "__main__":
    test_specific_case()
    # Run a few examples
    test_abstract_method_error_invalid_methodtype("test")
    test_abstract_method_error_invalid_methodtype("invalid")