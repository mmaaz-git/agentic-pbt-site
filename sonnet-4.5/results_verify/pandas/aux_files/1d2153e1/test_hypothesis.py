from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(
    st.lists(
        st.integers(min_value=-100, max_value=100),
        min_size=2,
        max_size=50,
    )
)
@settings(max_examples=100)
def test_argmin_argmax_consistency(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    try:
        assert sparse.argmin() == np.argmin(arr)
        assert sparse.argmax() == np.argmax(arr)
    except Exception as e:
        print(f"Failed on input: {values}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    # Test the specific failing case first
    print("Testing specific failing case: [0, 0]")
    try:
        test_argmin_argmax_consistency([0, 0])
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")

    print("\nRunning hypothesis tests...")
    test_argmin_argmax_consistency()