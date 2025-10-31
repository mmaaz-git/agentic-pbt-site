import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.backends.chunks import build_grid_chunks

@given(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=1, max_value=100)
)
def test_build_grid_chunks_sum_equals_size(size, chunk_size):
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    assert sum(chunks) == size, \
        f"sum(chunks)={sum(chunks)} != size={size} for chunks={chunks}"

# Run the test
test_build_grid_chunks_sum_equals_size()