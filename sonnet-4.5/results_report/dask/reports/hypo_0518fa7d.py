from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.arrow import ArrowORCEngine

@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=200)
def test_aggregate_files_with_none_stripes(split_stripes_val):
    parts = [[("file1.orc", None)], [("file2.orc", None)]]

    result = ArrowORCEngine._aggregate_files(
        aggregate_files=True,
        split_stripes=split_stripes_val,
        parts=parts
    )

    assert result is not None

# Run the test
if __name__ == "__main__":
    test_aggregate_files_with_none_stripes()