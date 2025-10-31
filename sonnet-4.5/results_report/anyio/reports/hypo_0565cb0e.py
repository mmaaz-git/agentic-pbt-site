from hypothesis import given, strategies as st
import math
from anyio.abc import CapacityLimiter


@given(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False).filter(lambda x: not x.is_integer()))
def test_capacity_limiter_accepts_float_tokens(value):
    limiter = CapacityLimiter(10)  # Use integer for initialization to avoid early failure
    limiter.total_tokens = value
    assert limiter.total_tokens == value

if __name__ == "__main__":
    test_capacity_limiter_accepts_float_tokens()