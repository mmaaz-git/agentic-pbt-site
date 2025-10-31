import pandas as pd
from hypothesis import given, strategies as st
from pandas.core.interchange.dataframe import PandasDataFrameXchg


@given(
    n_rows=st.integers(min_value=1, max_value=100),
    n_chunks=st.integers(min_value=2, max_value=20)
)
def test_get_chunks_no_empty_chunks(n_rows, n_chunks):
    df = pd.DataFrame({'a': range(n_rows)})
    xchg_df = PandasDataFrameXchg(df)
    chunks = list(xchg_df.get_chunks(n_chunks))

    for i, chunk in enumerate(chunks):
        assert chunk.num_rows() > 0, f"Chunk {i} is empty"

# Run the test
if __name__ == "__main__":
    test_get_chunks_no_empty_chunks()