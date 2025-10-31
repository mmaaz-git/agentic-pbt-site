import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
import dask.utils

@given(st.integers(min_value=0, max_value=2**60))
@settings(max_examples=500)
def test_format_bytes_length_claim(n):
    formatted = dask.utils.format_bytes(n)
    assert len(formatted) <= 10, f"format_bytes({n}) = {formatted!r} has length {len(formatted)} > 10"

if __name__ == "__main__":
    test_format_bytes_length_claim()