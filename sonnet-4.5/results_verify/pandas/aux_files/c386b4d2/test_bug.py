from hypothesis import given, strategies as st, settings
import pandas.errors
import pytest


class DummyClass:
    pass


@given(st.text(min_size=1).filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
@settings(max_examples=100)
def test_abstractmethoderror_invalid_methodtype_message(invalid_methodtype):
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    with pytest.raises(ValueError) as exc_info:
        pandas.errors.AbstractMethodError(DummyClass(), methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    for valid_type in valid_types:
        assert valid_type in error_message

    assert invalid_methodtype in error_message

    msg_start = error_message.split(',')[0]
    assert invalid_methodtype not in msg_start


# Run the test
if __name__ == "__main__":
    test_abstractmethoderror_invalid_methodtype_message("0")
    print("Test passed with methodtype='0'")