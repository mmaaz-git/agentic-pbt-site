import math
from hypothesis import given, strategies as st
import anyio


@given(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False).filter(lambda x: not x.is_integer()))
def test_capacity_limiter_accepts_float_tokens(total_tokens):
    limiter = anyio.CapacityLimiter(total_tokens)
    assert limiter.total_tokens == total_tokens

# Run the test
if __name__ == "__main__":
    test_capacity_limiter_accepts_float_tokens()