from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_argmin_argmax_no_crash(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    argmin_result = sparse.argmin()
    argmax_result = sparse.argmax()

    assert argmin_result == arr.argmin()
    assert argmax_result == arr.argmax()

# Run the test
if __name__ == "__main__":
    test_argmin_argmax_no_crash()