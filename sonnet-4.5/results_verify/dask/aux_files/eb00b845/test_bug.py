import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from hypothesis import given, strategies as st, example
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@example(1125894277343089729)  # The specific failing case
def test_format_bytes_output_length_invariant(n):
    result = format_bytes(n)
    print(f"Testing n={n}: format_bytes({n}) = '{result}' (length {len(result)})")
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"

if __name__ == "__main__":
    # First test the specific failing case
    n = 1125894277343089729
    result = format_bytes(n)
    print(f"\nSpecific test case:")
    print(f"  n = {n}")
    print(f"  n < 2**60 = {n < 2**60}")
    print(f"  format_bytes({n}) = '{result}'")
    print(f"  Length of result = {len(result)}")
    print(f"  Expected length <= 10, actual length = {len(result)}")

    # Run hypothesis test
    print("\nRunning Hypothesis test...")
    try:
        test_format_bytes_output_length_invariant()
    except AssertionError as e:
        print(f"Test failed: {e}")