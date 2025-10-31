from hypothesis import given, strategies as st, example
import pandas.errors as pd_errors
import pytest


class DummyClass:
    pass


@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"} and x != ""))
@example("invalid_type")  # Provide explicit example
def test_abstract_method_error_invalid_methodtype_message(invalid_methodtype):
    dummy = DummyClass()
    with pytest.raises(ValueError) as exc_info:
        pd_errors.AbstractMethodError(dummy, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    # Check that the error message format is correct
    # It should be "methodtype must be one of {valid_types}, got {invalid_methodtype} instead."
    # But the bug causes it to be "methodtype must be one of {invalid_methodtype}, got {valid_types} instead."

    # This assertion will fail due to the bug - the message has the arguments swapped
    expected_pattern = f"methodtype must be one of"
    assert expected_pattern in error_message

    # The bug: it says "must be one of invalid_type" instead of "must be one of {valid_types}"
    if f"one of {invalid_methodtype}" in error_message:
        print(f"BUG DETECTED: Error message incorrectly says 'must be one of {invalid_methodtype}'")
        print(f"Actual error message: {error_message}")
        assert False, f"Bug: error message has swapped arguments - says 'must be one of {invalid_methodtype}' instead of listing valid types"


if __name__ == "__main__":
    # Run the test with hypothesis
    test_abstract_method_error_invalid_methodtype_message()