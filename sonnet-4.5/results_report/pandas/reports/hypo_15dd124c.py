from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_cumsum_does_not_crash(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    cumsum_result = sparse.cumsum()
    assert len(cumsum_result) == len(sparse)

# Run the test
if __name__ == "__main__":
    test_cumsum_does_not_crash()