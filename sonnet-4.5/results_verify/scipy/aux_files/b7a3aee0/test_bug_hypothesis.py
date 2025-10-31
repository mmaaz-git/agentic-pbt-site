from hypothesis import given, strategies as st
import pandas as pd
import xarray.indexes as xr_indexes

@given(st.integers(min_value=-100, max_value=100))
def test_pandas_index_roll_on_empty_index(shift):
    """
    Property: roll should work on empty indexes without crashing.
    Rolling an empty index by any amount should return an empty index.
    """
    empty_pd_idx = pd.Index([])
    idx = xr_indexes.PandasIndex(empty_pd_idx, dim='x')

    result = idx.roll({'x': shift})

    assert len(result.index) == 0
    assert result.dim == idx.dim

# Run the test
if __name__ == "__main__":
    test_pandas_index_roll_on_empty_index()