import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')
from hypothesis import given, strategies as st, settings
from anyio._core._synchronization import CapacityLimiterAdapter

@given(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=10)
def test_capacity_limiter_accepts_float_total_tokens(value):
    limiter = CapacityLimiterAdapter(total_tokens=10)
    try:
        limiter.total_tokens = value
        assert limiter.total_tokens == value
        print(f"✓ Value {value} worked")
    except TypeError as e:
        print(f"✗ Value {value} failed: {e}")

print("Running Hypothesis test with various float values:")
test_capacity_limiter_accepts_float_total_tokens()
