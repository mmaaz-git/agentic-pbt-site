from hypothesis import given, settings, strategies as st
from xarray.backends.chunks import build_grid_chunks

@settings(max_examples=500)
@given(
    size=st.integers(min_value=1, max_value=10000),
    chunk_size=st.integers(min_value=1, max_value=1000)
)
def test_build_grid_chunks_sum_invariant(size, chunk_size):
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    assert sum(chunks) == size

if __name__ == "__main__":
    test_build_grid_chunks_sum_invariant()