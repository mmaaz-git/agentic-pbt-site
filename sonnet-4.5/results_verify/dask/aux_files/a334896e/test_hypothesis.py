import dask.dataframe as dd
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

@given(
    data_frames([
        column('a', dtype=int),
        column('b', dtype=float),
        column('c', dtype=str),
    ], index=range_indexes(min_size=1, max_size=100))
)
def test_from_pandas_compute_roundtrip(pdf):
    ddf = dd.from_pandas(pdf, npartitions=2)
    result = ddf.compute()
    pd.testing.assert_frame_equal(result, pdf)

if __name__ == "__main__":
    test_from_pandas_compute_roundtrip()