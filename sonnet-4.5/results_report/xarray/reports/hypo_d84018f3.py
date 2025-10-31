from hypothesis import given, strategies as st, settings
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100)
def test_build_grid_chunks_sum(size, chunk_size):
    chunks = build_grid_chunks(size, chunk_size)

    assert sum(chunks) == size, \
        f"Sum of chunks {sum(chunks)} != size {size} (chunks={chunks}, chunk_size={chunk_size})"

    assert all(c > 0 for c in chunks), \
        f"All chunks should be positive, got {chunks}"

if __name__ == "__main__":
    test_build_grid_chunks_sum()