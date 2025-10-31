from hypothesis import given, strategies as st, settings
from decimal import Decimal
import json
from pydantic.deprecated.json import decimal_encoder


@given(st.decimals(allow_nan=False, allow_infinity=False))
@settings(max_examples=1000)
def test_decimal_encoder_round_trip(dec):
    encoded = decimal_encoder(dec)
    assert isinstance(encoded, (int, float))

    json_str = json.dumps(encoded)
    decoded = json.loads(json_str)

    assert Decimal(str(decoded)) == dec

if __name__ == "__main__":
    test_decimal_encoder_round_trip()