import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray
from hypothesis import given, strategies as st, settings


@given(
    lists=st.lists(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=10),
        min_size=1, max_size=20
    ),
    start=st.integers(min_value=-15, max_value=15),
    stop=st.integers(min_value=-15, max_value=15) | st.none(),
    step=st.integers(min_value=1, max_value=3) | st.none()
)
@settings(max_examples=500)
def test_list_accessor_slice_consistency(lists, start, stop, step):
    pa_array = pa.array(lists, type=pa.list_(pa.int64()))
    arr = ArrowExtensionArray(pa_array)
    s = pd.Series(arr)

    try:
        sliced = s.list[start:stop:step]

        for i in range(len(s)):
            original_list = lists[i]
            sliced_value = sliced.iloc[i]
            expected_slice = original_list[start:stop:step]

            assert len(sliced_value) == len(expected_slice)
    except NotImplementedError:
        pass

# Run the test
if __name__ == "__main__":
    test_list_accessor_slice_consistency()
    print("Test completed successfully!")