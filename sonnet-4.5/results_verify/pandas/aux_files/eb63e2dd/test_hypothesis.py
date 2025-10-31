from hypothesis import given, strategies as st
import pandas.errors as pe


class DummyClass:
    pass


@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_invalid_methodtype_message(invalid_methodtype):
    try:
        pe.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        valid_types = {"method", "classmethod", "staticmethod", "property"}

        assert error_msg.index(str(valid_types)) < error_msg.index(invalid_methodtype), \
            f"Valid types should appear before invalid input in error message, got: {error_msg}"

# Run the test
test_abstract_method_error_invalid_methodtype_message()