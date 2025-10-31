from decimal import Decimal
from hypothesis import given, strategies as st, example
from pydantic.deprecated.json import decimal_encoder


@given(st.decimals(allow_nan=False, allow_infinity=False))
@example(Decimal('1.0'))
@example(Decimal('42.00'))
def test_decimal_encoder_integer_values_encode_as_int(dec):
    encoded = decimal_encoder(dec)
    is_integer_value = dec == dec.to_integral_value()

    if is_integer_value:
        assert isinstance(encoded, int), (
            f"Integer-valued Decimal {dec} (exponent={dec.as_tuple().exponent}) "
            f"should encode as int, got {type(encoded).__name__}"
        )

if __name__ == "__main__":
    # Run the test
    test_decimal_encoder_integer_values_encode_as_int()