from decimal import Decimal
from hypothesis import given, strategies as st
from fastapi.encoders import decimal_encoder

@given(st.decimals(allow_nan=False, allow_infinity=False))
def test_decimal_encoder_round_trip(dec_value):
    encoded = decimal_encoder(dec_value)
    decoded = Decimal(str(encoded))
    assert decoded == dec_value, f"Round-trip failed: {dec_value} -> {encoded} -> {decoded}"

if __name__ == "__main__":
    test_decimal_encoder_round_trip()