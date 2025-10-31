import pandas.errors as pd_errors
from hypothesis import given, strategies as st, assume


@given(st.text(min_size=1))
def test_abstractmethoderror_error_message_property(invalid_methodtype):
    valid_types = {"method", "classmethod", "staticmethod", "property"}
    assume(invalid_methodtype not in valid_types)

    class DummyClass:
        pass

    try:
        pd_errors.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        print(f"Testing with invalid_methodtype='{invalid_methodtype}'")
        print(f"Error message: {error_msg}")
        assert str(valid_types) in error_msg, f"Error should mention valid types, got: {error_msg}"
        assert f"got {invalid_methodtype}" in error_msg, f"Error should mention invalid input, got: {error_msg}"

if __name__ == "__main__":
    test_abstractmethoderror_error_message_property()