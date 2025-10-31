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
    # Test with the specific failing input mentioned
    arr = np.array([0, 0, 0, 0, 0])
    indices = []
    print(f"Testing with arr={arr}, indices={indices}")

    try:
        indices_arr = np.array(indices)
        print(f"indices_arr type: {type(indices_arr)}, dtype: {indices_arr.dtype}, shape: {indices_arr.shape}")
        result = indexers.check_array_indexer(arr, indices_arr)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")