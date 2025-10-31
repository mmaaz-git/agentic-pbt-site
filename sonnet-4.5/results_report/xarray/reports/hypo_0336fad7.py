import warnings
from hypothesis import given, strategies as st
from pydantic.deprecated.json import pydantic_encoder
import json

warnings.filterwarnings('ignore', category=DeprecationWarning)

@given(st.binary(min_size=1, max_size=100))
def test_pydantic_encoder_bytes(b):
    result = json.dumps(b, default=pydantic_encoder)
    assert isinstance(result, str)

if __name__ == "__main__":
    test_pydantic_encoder_bytes()