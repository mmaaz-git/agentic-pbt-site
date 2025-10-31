from pandas.arrays import SparseArray
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50)
)
@settings(max_examples=500)
def test_argmin_argmax_values(data):
    arr = SparseArray(data)

    argmin_idx = arr.argmin()
    argmax_idx = arr.argmax()

    assert arr[argmin_idx] == arr.min()
    assert arr[argmax_idx] == arr.max()

if __name__ == "__main__":
    test_argmin_argmax_values()