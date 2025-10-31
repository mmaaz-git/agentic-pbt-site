import pandas as pd
from hypothesis import given, strategies as st


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=2, max_value=10),
)
def test_chunk_sizes_reasonable(nrows, n_chunks):
    df = pd.DataFrame({'a': list(range(nrows))})
    interchange_df = df.__dataframe__()
    chunks = list(interchange_df.get_chunks(n_chunks=n_chunks))

    chunk_sizes = [chunk.num_rows() for chunk in chunks]

    for size in chunk_sizes:
        assert size > 0, f"Chunk has {size} rows (should be > 0)"


if __name__ == "__main__":
    test_chunk_sizes_reasonable()