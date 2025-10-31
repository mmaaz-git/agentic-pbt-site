#!/usr/bin/env python3
"""Run the hypothesis test from the bug report."""

from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe import PandasDataFrameXchg

@given(
    n_rows=st.integers(min_value=1, max_value=100),
    n_chunks=st.integers(min_value=2, max_value=200),
)
@settings(max_examples=50, deadline=None)
def test_no_empty_chunks(n_rows, n_chunks):
    df = pd.DataFrame(np.random.randn(n_rows, 3))
    xchg = PandasDataFrameXchg(df, allow_copy=True)
    chunks = list(xchg.get_chunks(n_chunks=n_chunks))

    empty_chunks = [c for c in chunks if c.num_rows() == 0]
    if len(empty_chunks) > 0:
        print(f"\nFailed with n_rows={n_rows}, n_chunks={n_chunks}")
        print(f"  Found {len(empty_chunks)} empty chunks out of {len(chunks)} total")
        assert False, f"Found {len(empty_chunks)} empty chunks"

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_no_empty_chunks()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis confirms the bug - empty chunks are created when n_chunks > n_rows")