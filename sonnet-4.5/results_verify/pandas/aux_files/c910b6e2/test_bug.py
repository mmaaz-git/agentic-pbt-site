import pandas as pd
from hypothesis import given, strategies as st
import pytest

# First, let's run the simple reproduction
print("=== Simple Reproduction ===")
try:
    error = pd.errors.AbstractMethodError(object(), methodtype="invalid_type")
except ValueError as e:
    print(f"Actual:   {e}")
    print(f"Expected: methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got invalid_type instead.")

print("\n=== Testing with different invalid values ===")
# Test with various invalid values
test_values = ["0", "invalid", "foo", "bar", "test123", ""]
for val in test_values:
    try:
        error = pd.errors.AbstractMethodError(object(), methodtype=val)
    except ValueError as e:
        print(f"Value '{val}': {e}")

print("\n=== Running Hypothesis test ===")
@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_message_parameters_not_swapped(invalid_methodtype):
    with pytest.raises(ValueError) as exc_info:
        pd.errors.AbstractMethodError(object(), methodtype=invalid_methodtype)

    error_msg = str(exc_info.value)

    # Check if the error message contains the invalid value after 'got'
    assert f"got {invalid_methodtype}" in error_msg, \
        f"Error message should say 'got {invalid_methodtype}', but got: {error_msg}"

    # Also check that valid values come after 'must be one of'
    assert "must be one of {" in error_msg or "must be one of method" not in error_msg, \
        f"Error message format is incorrect: {error_msg}"

# Run a simple test
print("Testing with '0':")
test_abstract_method_error_message_parameters_not_swapped("0")