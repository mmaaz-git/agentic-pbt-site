from hypothesis import given, strategies as st
import math
from anyio import CapacityLimiter

@given(st.floats(min_value=1.0, max_value=1000.0).filter(
    lambda x: x != math.inf and not (isinstance(x, float) and x == int(x)) and not math.isnan(x)
))
def test_capacity_limiter_type_contract(total_tokens):
    """
    Test that CapacityLimiter accepts all non-negative floats >= 1 as per type annotation.
    The type annotation says 'float', so all floats >= 1 should be valid.
    """
    limiter = CapacityLimiter(total_tokens)
    assert limiter.total_tokens == total_tokens

if __name__ == "__main__":
    test_capacity_limiter_type_contract()