import pandas as pd
import io
from hypothesis import given, strategies as st, settings, example
import hypothesis.extra.pandas as pdst


@given(
    df=pdst.data_frames(
        columns=[
            pdst.column('a', dtype=int),
            pdst.column('b', dtype=float),
            pdst.column('c', dtype=str),
        ],
        rows=st.tuples(
            st.integers(min_value=-1000, max_value=1000),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=20)
        )
    )
)
@settings(max_examples=100, deadline=10000)
def test_chunking_equivalence(df):
    if len(df) == 0:
        return

    csv_string = df.to_csv(index=False)

    full_read = pd.read_csv(io.StringIO(csv_string))

    chunksize = max(1, len(df) // 3)
    chunks = []
    for chunk in pd.read_csv(io.StringIO(csv_string), chunksize=chunksize):
        chunks.append(chunk)

    chunked_read = pd.concat(chunks, ignore_index=True)

    try:
        pd.testing.assert_frame_equal(full_read, chunked_read)
        print(f"PASS: df shape={df.shape}")
    except AssertionError as e:
        print(f"FAIL: df shape={df.shape}")
        print(f"CSV content:\n{csv_string}")
        print(f"Full read:\n{full_read}")
        print(f"Chunked read:\n{chunked_read}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    # Run the test
    test_chunking_equivalence()