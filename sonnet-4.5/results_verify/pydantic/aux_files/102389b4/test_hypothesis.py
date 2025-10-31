import numpy as np
from pandas.core.arrays import SparseArray
from hypothesis import given, strategies as st, settings


@given(
    data=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=5, max_size=50),
    old_fill=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    new_fill=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_sparsearray_change_fill_value(data, old_fill, new_fill):
    sparse = SparseArray(data, fill_value=old_fill)
    original_dense = sparse.to_dense()

    new_sparse = SparseArray(sparse, fill_value=new_fill)

    assert np.allclose(new_sparse.to_dense(), original_dense, equal_nan=True, rtol=1e-10)

if __name__ == "__main__":
    # Run the test
    test_sparsearray_change_fill_value()
    print("All tests passed!")