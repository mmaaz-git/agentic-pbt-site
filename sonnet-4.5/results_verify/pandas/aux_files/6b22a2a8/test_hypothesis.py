import numpy as np
from pandas.core.sparse.api import SparseArray
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_astype_returns_sparsearray(data):
    sparse = SparseArray(data, dtype=np.int64)
    result = sparse.astype(np.float64)
    assert isinstance(result, SparseArray), \
        f"astype() should return SparseArray, got {type(result)}"

# Run the test
if __name__ == "__main__":
    test_astype_returns_sparsearray()