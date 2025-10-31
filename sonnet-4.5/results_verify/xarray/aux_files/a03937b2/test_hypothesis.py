from hypothesis import given, strategies as st, settings
import pandas as pd
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env')
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
    try:
        rolled = index.roll({dim: shift})
        print(f"SUCCESS: Rolling index of length {len(index.index)} by {shift}")
    except ZeroDivisionError as e:
        print(f"FAILURE: ZeroDivisionError for index of length {len(index.index)} with shift {shift}")
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    test_pandasindex_roll_no_crash()