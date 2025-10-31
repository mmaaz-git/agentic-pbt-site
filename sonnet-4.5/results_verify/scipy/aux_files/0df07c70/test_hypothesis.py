from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.sparse as sp


@given(
    n=st.integers(min_value=1, max_value=20),
    k=st.integers(min_value=-5, max_value=5)
)
@settings(max_examples=200)
def test_eye_matches_dense(n, k):
    sparse_eye = sp.eye_array(n, k=k).toarray()
    dense_eye = np.eye(n, k=k)

    np.testing.assert_array_equal(
        sparse_eye,
        dense_eye,
        err_msg="eye_array doesn't match numpy.eye"
    )

# Run the test
if __name__ == "__main__":
    try:
        test_eye_matches_dense()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")