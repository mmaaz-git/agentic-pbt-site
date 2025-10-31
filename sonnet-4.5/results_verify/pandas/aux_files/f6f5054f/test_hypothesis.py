import pandas.errors
from hypothesis import given, strategies as st, example
import pytest


@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
@example("0")
@example("invalid")
def test_abstract_method_error_message_format(invalid_methodtype):
    with pytest.raises(ValueError) as exc_info:
        pandas.errors.AbstractMethodError(object(), methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    valid_types_set = {"method", "classmethod", "staticmethod", "property"}

    for valid_type in valid_types_set:
        assert valid_type in error_message, (
            f"Valid type '{valid_type}' should appear in error message, "
            f"but error message is: '{error_message}'"
        )

    parts_after_got = error_message.split("got")
    assert len(parts_after_got) > 1, "Error message should contain 'got'"

    got_part = parts_after_got[1]
    assert invalid_methodtype in got_part, (
        f"The invalid input '{invalid_methodtype}' should appear after 'got' "
        f"in the error message, but the part after 'got' is: '{got_part}'"
    )