import pandas.errors
from hypothesis import given, strategies as st

class DummyClass:
    pass

@given(st.text(min_size=1).filter(lambda x: x not in {'method', 'classmethod', 'staticmethod', 'property'}))
def test_abstract_method_error_invalid_methodtype_message(invalid_methodtype):
    """
    Property: When AbstractMethodError is given an invalid methodtype,
    the error message should mention the invalid value and the valid options.
    Specifically, the error message should contain:
    1. The invalid methodtype value provided
    2. The set of valid options
    And the structure should be: "must be one of {valid_options}, got {invalid_value}"
    """
    valid_types = {'method', 'classmethod', 'staticmethod', 'property'}

    try:
        pandas.errors.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)
        assert False, f"Should have raised ValueError for invalid methodtype {invalid_methodtype}"
    except ValueError as e:
        error_msg = str(e)

        assert invalid_methodtype in error_msg, \
            f"Error message should mention the invalid value '{invalid_methodtype}', but got: {error_msg}"

        assert all(vtype in error_msg for vtype in valid_types), \
            f"Error message should mention all valid types {valid_types}, but got: {error_msg}"

        idx_invalid = error_msg.find(invalid_methodtype)
        idx_valid_set_start = error_msg.find('{')

        assert idx_valid_set_start < idx_invalid, \
            f"Error message structure is wrong: valid types should come before invalid value. Got: {error_msg}"

if __name__ == "__main__":
    test_abstract_method_error_invalid_methodtype_message()