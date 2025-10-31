from hypothesis import given, settings, strategies as st
from anyio._core._synchronization import CapacityLimiter


@given(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_capacity_limiter_accepts_float_tokens(tokens):
    limiter = CapacityLimiter(tokens)
    assert limiter.total_tokens == tokens


if __name__ == "__main__":
    test_capacity_limiter_accepts_float_tokens()