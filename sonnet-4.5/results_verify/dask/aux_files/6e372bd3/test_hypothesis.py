#!/usr/bin/env python3
"""Property-based test for dask.dataframe.io.orc.read_orc"""

import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings
import dask.dataframe as dd
import traceback

@given(
    n_rows=st.integers(min_value=20, max_value=100),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=5)  # Reduced for testing
def test_columns_mutation_when_index_not_in_columns(n_rows, seed):
    """
    Property: Should be able to read ORC with index column not in columns list.
    The index column should be automatically included in the read.
    """
    tmp = tempfile.mkdtemp()

    try:
        np.random.seed(seed)
        df_pandas = pd.DataFrame({
            "a": np.random.randint(0, 100, size=n_rows, dtype=np.int32),
            "b": np.random.randint(0, 100, size=n_rows, dtype=np.int32),
            "c": np.random.randint(0, 100, size=n_rows, dtype=np.int32),
        })

        df_dask = dd.from_pandas(df_pandas, npartitions=2)
        df_dask.to_orc(tmp, write_index=False)

        columns_to_read = ["a", "b"]
        index_col = "c"

        df_read = dd.read_orc(tmp, columns=columns_to_read, index=index_col)
        result = df_read.compute()

        assert list(result.columns) == ["a", "b"]
        assert result.index.name == "c"

    except Exception as e:
        print(f"Failed with n_rows={n_rows}, seed={seed}")
        print(f"Error: {e}")
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    print("=== Running Property-Based Test ===")
    try:
        test_columns_mutation_when_index_not_in_columns()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()