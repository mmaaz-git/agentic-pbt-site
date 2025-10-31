from anyio import CapacityLimiter
from hypothesis import given, strategies as st
import math

@given(st.sampled_from([math.inf, float('inf'), float('infinity')]))
def test_capacity_limiter_accepts_all_infinity_representations(inf_value):
    limiter = CapacityLimiter(inf_value)
    assert limiter.total_tokens == inf_value

if __name__ == "__main__":
    test_capacity_limiter_accepts_all_infinity_representations()