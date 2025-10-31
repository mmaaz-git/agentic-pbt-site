import pandas as pd
from pandas.core.interchange.dataframe import PandasDataFrameXchg

def test_get_chunks_direct():
    # Test case 1: n_rows=1, n_chunks=2
    df = pd.DataFrame({'a': [1]})
    xchg_df = PandasDataFrameXchg(df)
    chunks = list(xchg_df.get_chunks(n_chunks=2))

    print("Test case 1: n_rows=1, n_chunks=2")
    for i, chunk in enumerate(chunks):
        num_rows = chunk.num_rows()
        print(f"  Chunk {i}: {num_rows} rows")
        if num_rows == 0:
            print(f"  ERROR: Chunk {i} is empty!")

    # Test case 2: n_rows=5, n_chunks=10
    df2 = pd.DataFrame({'a': range(5)})
    xchg_df2 = PandasDataFrameXchg(df2)
    chunks2 = list(xchg_df2.get_chunks(n_chunks=10))

    print("\nTest case 2: n_rows=5, n_chunks=10")
    empty_count = 0
    for i, chunk in enumerate(chunks2):
        num_rows = chunk.num_rows()
        print(f"  Chunk {i}: {num_rows} rows")
        if num_rows == 0:
            empty_count += 1
    print(f"  Total empty chunks: {empty_count}")

if __name__ == "__main__":
    test_get_chunks_direct()