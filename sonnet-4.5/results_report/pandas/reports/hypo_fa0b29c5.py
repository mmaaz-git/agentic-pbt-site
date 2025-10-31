from hypothesis import given, strategies as st
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(
    st.lists(
        st.integers(min_value=-100, max_value=100),
        min_size=2,
        max_size=50,
    )
)
def test_argmin_argmax_consistency(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    assert sparse.argmin() == np.argmin(arr)
    assert sparse.argmax() == np.argmax(arr)

if __name__ == "__main__":
    test_argmin_argmax_consistency()