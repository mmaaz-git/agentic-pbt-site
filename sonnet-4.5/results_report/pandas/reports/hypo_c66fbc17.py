from pandas.arrays import SparseArray
import numpy as np
from hypothesis import given, strategies as st, assume

@given(
    st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=10),
    st.lists(st.integers(min_value=0, max_value=10), min_size=3, max_size=10)
)
def test_sparse_operation_invariant(values1, values2):
    assume(len(values1) == len(values2))
    assume(all(v != 0 for v in values1))  # Ensure left has ngaps==0

    left = SparseArray(values1, fill_value=0)
    right = SparseArray(values2, fill_value=0)
    result = left - right

    # Invariant: sp_values should not contain fill_value
    assert not np.any(result.sp_values == result.fill_value), \
        f"sp_values {result.sp_values} contains fill_value {result.fill_value}"

# Run the test
if __name__ == "__main__":
    test_sparse_operation_invariant()