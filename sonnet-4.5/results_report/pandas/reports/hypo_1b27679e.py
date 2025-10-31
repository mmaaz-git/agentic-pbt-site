from hypothesis import given, strategies as st, settings
from pandas.core.arrays.sparse import SparseArray

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=100)
)
@settings(max_examples=300)
def test_argmin_argmax_consistency(data, fill_value):
    arr = SparseArray(data, fill_value=fill_value)
    dense = arr.to_dense()

    if len(arr) > 0:
        assert arr[arr.argmin()] == dense[dense.argmin()]
        assert arr[arr.argmax()] == dense[dense.argmax()]

# Run the test
if __name__ == "__main__":
    test_argmin_argmax_consistency()