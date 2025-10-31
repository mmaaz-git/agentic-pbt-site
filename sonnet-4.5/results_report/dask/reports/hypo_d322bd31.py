from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from dask.dataframe.utils import check_matching_columns

@given(st.lists(st.floats(allow_nan=False), min_size=1, max_size=5))
def test_check_matching_columns_nan_vs_zero(cols):
    meta_cols = [0] + cols
    actual_cols = [float('nan')] + cols

    meta = pd.DataFrame(columns=meta_cols)
    actual = pd.DataFrame(columns=actual_cols)

    try:
        check_matching_columns(meta, actual)
        assert False, f"Should raise ValueError for NaN vs 0 column mismatch"
    except ValueError:
        pass

if __name__ == "__main__":
    test_check_matching_columns_nan_vs_zero()