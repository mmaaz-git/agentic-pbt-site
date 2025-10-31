from decimal import Decimal
from hypothesis import given, strategies as st, assume, settings
from django.db.models.fields import DecimalField
from django.core import exceptions

@given(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=0, max_value=10)
)
@settings(max_examples=200)
def test_decimal_field_float_vs_decimal_consistency(float_val, max_digits, decimal_places):
    assume(decimal_places < max_digits)

    field = DecimalField(max_digits=max_digits, decimal_places=decimal_places)
    decimal_val = Decimal(str(float_val))

    try:
        result_float = field.to_python(float_val)
        result_decimal = field.to_python(decimal_val)

        assert result_float == result_decimal, (
            f"Inconsistent results for same value: "
            f"float({float_val}) -> {result_float}, "
            f"Decimal('{decimal_val}') -> {result_decimal}"
        )
    except exceptions.ValidationError:
        pass

if __name__ == "__main__":
    test_decimal_field_float_vs_decimal_consistency()