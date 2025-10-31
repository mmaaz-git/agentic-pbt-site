from pandas.arrays import SparseArray
import numpy as np
from hypothesis import given, strategies as st, assume

@given(
    st.lists(st.integers(min_value=-10, max_value=10), min_size=2, max_size=10),
    st.lists(st.integers(min_value=-10, max_value=10), min_size=2, max_size=10),
    st.integers(min_value=-10, max_value=10),
    st.integers(min_value=-10, max_value=10)
)
def test_concat_preserves_values(vals1, vals2, fill1, fill2):
    assume(fill1 != fill2)
    arr1 = SparseArray(vals1, fill_value=fill1)
    arr2 = SparseArray(vals2, fill_value=fill2)

    result = SparseArray._concat_same_type([arr1, arr2])
    expected = np.concatenate([arr1.to_dense(), arr2.to_dense()])

    np.testing.assert_array_equal(result.to_dense(), expected)

if __name__ == "__main__":
    test_concat_preserves_values()