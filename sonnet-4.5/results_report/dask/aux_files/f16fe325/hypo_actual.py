import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings


@settings(max_examples=50, deadline=10000)
@given(
    num_rows=st.integers(min_value=0, max_value=50),
)
def test_orc_empty_dataframe_round_trip(num_rows):
    df_pandas = pd.DataFrame({
        'col_a': list(range(num_rows)),
        'col_b': [f'val_{i}' for i in range(num_rows)],
    })
    df = dd.from_pandas(df_pandas, npartitions=1)

    tmpdir = tempfile.mkdtemp()
    try:
        dd.to_orc(df, tmpdir, write_index=False)
        df_read = dd.read_orc(tmpdir)
        df_result = df_read.compute()

        assert len(df_result) == num_rows, \
            f"Row count mismatch: {len(df_result)} != {num_rows}"

        if num_rows == 0:
            assert list(df_result.columns) == ['col_a', 'col_b'], \
                "Column names should be preserved even for empty dataframes"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    # Run the hypothesis test
    test_orc_empty_dataframe_round_trip()
