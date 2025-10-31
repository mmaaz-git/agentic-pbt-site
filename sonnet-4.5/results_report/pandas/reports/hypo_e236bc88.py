import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import pyarrow as pa
from hypothesis import given, strategies as st
from pandas.core.arrays.arrow import ArrowExtensionArray

@given(
    st.lists(st.integers(min_value=-1000, max_value=1000) | st.none(), min_size=1, max_size=50),
    st.integers(min_value=-1000, max_value=1000)
)
def test_arrow_array_fillna_removes_all_nulls(data, fill_value):
    arr = ArrowExtensionArray(pa.array(data))
    filled = arr.fillna(fill_value)

    filled_list = filled.tolist()
    for val in filled_list:
        assert val is not None and not pd.isna(val)

if __name__ == "__main__":
    test_arrow_array_fillna_removes_all_nulls()