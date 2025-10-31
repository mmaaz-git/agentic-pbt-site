#!/usr/bin/env python3
import decimal
from hypothesis import given, strategies as st, assume, settings
from django.db.models.fields import DecimalField
from django.core.exceptions import ValidationError

@given(st.floats(allow_nan=False, allow_infinity=False), st.integers(min_value=1, max_value=20), st.integers(min_value=0, max_value=10))
@settings(max_examples=100)
def test_decimalfield_float_vs_string_conversion(float_value, max_digits, decimal_places):
    assume(decimal_places <= max_digits)

    field = DecimalField(max_digits=max_digits, decimal_places=decimal_places)

    try:
        result_from_float = field.to_python(float_value)
        result_from_string = field.to_python(str(float_value))

        if result_from_float != result_from_string:
            print(f"Found mismatch: float_value={float_value}, max_digits={max_digits}, decimal_places={decimal_places}")
            print(f"  Float result: {result_from_float}")
            print(f"  String result: {result_from_string}")

        assert result_from_float == result_from_string
    except (ValidationError, decimal.InvalidOperation):
        pass

# Run the test
print("Running hypothesis test...")
try:
    test_decimalfield_float_vs_string_conversion()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")