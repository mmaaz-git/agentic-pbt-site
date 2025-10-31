from hypothesis import given, strategies as st, assume
from hypothesis.extra import numpy as npst
import numpy as np
from pandas.api import indexers

@given(
    npst.arrays(dtype=np.int64, shape=(5,)),
    st.lists(st.integers(min_value=0, max_value=4), min_size=0, max_size=10)
)
def test_check_array_indexer_basic(arr, indices):
    indices_arr = np.array(indices)
    result = indexers.check_array_indexer(arr, indices_arr)
    assert len(result) == len(indices)

if __name__ == "__main__":
    # Run the test
    test_check_array_indexer_basic()