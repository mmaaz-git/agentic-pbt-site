from hypothesis import given, strategies as st
import pandas.errors
import pytest

@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_invalid_methodtype_message_clarity(invalid_methodtype):
    with pytest.raises(ValueError) as exc_info:
        pandas.errors.AbstractMethodError(object(), methodtype=invalid_methodtype)

    error_msg = str(exc_info.value)
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    for valid_type in valid_types:
        parts = error_msg.split(", got")
        if len(parts) == 2:
            assert valid_type not in parts[1], \
                f"Valid type '{valid_type}' should not appear in 'got X' part"

if __name__ == "__main__":
    test_abstract_method_error_invalid_methodtype_message_clarity()