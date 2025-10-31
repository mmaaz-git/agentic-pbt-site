#!/usr/bin/env python3
import dask.dataframe as dd
import pandas as pd
from hypothesis import given, strategies as st, settings, example
from hypothesis.extra.pandas import data_frames, column, range_indexes

@given(
    data_frames([
        column('a', dtype=int),
        column('b', dtype=float),
        column('c', dtype=str),
    ], index=range_indexes(min_size=1, max_size=100))
)
@example(pd.DataFrame({'a': [0], 'b': [0.0], 'c': ['\ud800']}))
@settings(max_examples=10)
def test_from_pandas_compute_roundtrip(pdf):
    print(f"Testing with DataFrame: {pdf.head()}")
    print(f"String column values: {pdf['c'].tolist() if 'c' in pdf else 'N/A'}")
    try:
        ddf = dd.from_pandas(pdf, npartitions=2)
        result = ddf.compute()
        pd.testing.assert_frame_equal(result, pdf)
        print("✓ Test passed")
    except Exception as e:
        print(f"✗ Test failed with error: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    test_from_pandas_compute_roundtrip()