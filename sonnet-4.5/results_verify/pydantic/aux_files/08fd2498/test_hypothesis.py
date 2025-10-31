from hypothesis import given, strategies as st, settings
from pydantic.deprecated.json import ENCODERS_BY_TYPE


@given(st.binary())
@settings(max_examples=500)
def test_bytes_encoder(b):
    encoder = ENCODERS_BY_TYPE[bytes]
    result = encoder(b)
    assert isinstance(result, str)

# Run the test
if __name__ == "__main__":
    test_bytes_encoder()