from decimal import Decimal
from hypothesis import given, strategies as st
from pydantic.deprecated.json import decimal_encoder
import warnings


@given(st.decimals(allow_nan=False, allow_infinity=False))
def test_decimal_encoder_round_trip(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoded = decimal_encoder(x)
        assert isinstance(encoded, (int, float))

        decoded = Decimal(str(encoded))

        assert decoded == x, f"Round-trip failed: {x} -> {encoded} -> {decoded}"

if __name__ == "__main__":
    test_decimal_encoder_round_trip()