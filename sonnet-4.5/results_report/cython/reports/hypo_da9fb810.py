import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.interchange.dataframe import PandasDataFrameXchg


@given(
    size=st.integers(min_value=1, max_value=100),
    n_chunks=st.integers(min_value=2, max_value=20)
)
@settings(max_examples=500)
def test_get_chunks_should_not_produce_empty_chunks(size, n_chunks):
    df = pd.DataFrame({'A': range(size)})
    interchange_obj = PandasDataFrameXchg(df)

    chunks = list(interchange_obj.get_chunks(n_chunks))

    for i, chunk in enumerate(chunks):
        if chunk.num_rows() == 0:
            raise AssertionError(
                f"Chunk {i} is empty! size={size}, n_chunks={n_chunks}, "
                f"chunk_sizes={[c.num_rows() for c in chunks]}"
            )


if __name__ == "__main__":
    # Run the test
    test_get_chunks_should_not_produce_empty_chunks()