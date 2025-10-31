from hypothesis import given, strategies as st, settings
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100)
def test_build_grid_chunks_preserves_size(size, chunk_size):
    result = build_grid_chunks(size, chunk_size)
    assert sum(result) == size, f"Chunks {result} sum to {sum(result)}, expected {size}"

if __name__ == "__main__":
    # Run the test and let Hypothesis find and report the failing example
    test_build_grid_chunks_preserves_size()