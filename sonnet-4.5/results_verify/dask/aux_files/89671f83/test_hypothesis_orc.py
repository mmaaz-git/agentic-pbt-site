from hypothesis import given, strategies as st, settings
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd


@given(
    st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=10),
    st.booleans(),
)
@settings(max_examples=100, deadline=None)
def test_orc_round_trip_preserves_index(data, write_index):
    tmpdir = tempfile.mkdtemp()
    try:
        df = pd.DataFrame({"a": data, "b": [x * 2 for x in data]})
        ddf = dd.from_pandas(df, npartitions=2)

        path = f"{tmpdir}/test_orc"
        ddf.to_orc(path, write_index=write_index)

        result = dd.read_orc(path)
        result_df = result.compute()

        if write_index:
            pd.testing.assert_frame_equal(result_df, df, check_dtype=False)
        else:
            pd.testing.assert_frame_equal(
                result_df.reset_index(drop=True), df, check_dtype=False
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    # Run the test
    test_orc_round_trip_preserves_index()
    print("All tests passed!")