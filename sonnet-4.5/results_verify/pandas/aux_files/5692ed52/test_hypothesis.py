from hypothesis import given, strategies as st, settings
import pandas.errors
import pytest


@given(
    methodtype=st.text().filter(
        lambda x: x not in {"method", "classmethod", "staticmethod", "property"}
    )
)
@settings(max_examples=10)  # Reduced for quick testing
def test_abstract_method_error_invalid_methodtype_raises(methodtype):
    """Test that invalid methodtypes raise ValueError with correct message."""
    class TestClass:
        pass

    instance = TestClass()

    with pytest.raises(ValueError) as exc_info:
        pandas.errors.AbstractMethodError(instance, methodtype=methodtype)

    error_msg = str(exc_info.value)
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    print(f"Testing methodtype={repr(methodtype)}")
    print(f"Error message: {error_msg}")

    assert "methodtype must be one of" in error_msg
    # The bug: the message has parameters swapped
    # It says "methodtype must be one of <invalid_value>, got <valid_types> instead"
    # Instead of "methodtype must be one of <valid_types>, got <invalid_value> instead"

    # This assertion will fail due to the bug
    try:
        assert f"got {methodtype}" in error_msg or repr(methodtype) in error_msg
        print("  ✓ Pass: Found invalid value after 'got'")
    except AssertionError:
        print(f"  ✗ FAIL: Expected 'got {methodtype}' but message is: {error_msg}")
        raise

# Run with specific test case from bug report
test_abstract_method_error_invalid_methodtype_raises('0')