from hypothesis import given, strategies as st, settings
import pandas as pd
from xarray.core.indexes import PandasIndex

@st.composite
def xarray_pandas_indexes_including_empty(draw):
    size = draw(st.integers(min_value=0, max_value=100))
    if size == 0:
        pd_index = pd.Index([])
    else:
        values = draw(st.lists(st.integers(), min_size=size, max_size=size))
        pd_index = pd.Index(values)
    dim_name = draw(st.text(min_size=1, max_size=10))
    return PandasIndex(pd_index, dim_name)

@settings(max_examples=200)
@given(xarray_pandas_indexes_including_empty(), st.integers(min_value=-100, max_value=100))
def test_pandasindex_roll_no_crash(index, shift):
    dim = index.dim
    rolled = index.roll({dim: shift})
    # The test passes if no exception is raised

if __name__ == "__main__":
    test_pandasindex_roll_no_crash()