#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import tempfile
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings
import traceback

@settings(max_examples=5)  # Reduced for testing
@given(
    nrows=st.integers(min_value=20, max_value=100),
    npartitions=st.integers(min_value=2, max_value=5),
)
def test_read_orc_columns_not_duplicated_across_partitions(nrows, npartitions):
    with tempfile.TemporaryDirectory() as tmpdir:
        data = pd.DataFrame({
            'a': range(nrows),
            'b': range(nrows, 2 * nrows),
            'c': range(2 * nrows, 3 * nrows),
        })
        df = dd.from_pandas(data, npartitions=npartitions)
        df.to_orc(tmpdir, write_index=False)

        result = dd.read_orc(tmpdir, columns=['b', 'c'], index='a')
        result_df = result.compute()

        assert 'a' not in result_df.columns
        assert list(result_df.columns) == ['b', 'c']
        assert result_df.index.name == 'a'

if __name__ == "__main__":
    try:
        test_read_orc_columns_not_duplicated_across_partitions()
        print("Hypothesis test PASSED!")
    except Exception as e:
        print(f"Hypothesis test FAILED: {e}")
        traceback.print_exc()