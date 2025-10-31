import numpy as np
from pandas.core.arrays.sparse import SparseArray
from hypothesis import given, strategies as st, settings

@settings(max_examples=100)
@given(
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=1, max_value=20)
)
def test_argmin_argmax_all_fill_values(fill_value, size):
    data = [fill_value] * size
    arr = SparseArray(data, fill_value=fill_value)
    dense = np.array(data)

    assert arr.argmin() == dense.argmin()
    assert arr.argmax() == dense.argmax()

if __name__ == "__main__":
    test_argmin_argmax_all_fill_values()