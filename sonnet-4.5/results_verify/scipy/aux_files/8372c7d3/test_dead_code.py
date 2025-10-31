from hypothesis import given, strategies as st
import numpy as np
from scipy.sparse.csgraph._laplacian import _laplacian_dense

@given(st.sampled_from([True, False]))
def test_laplacian_dense_dtype_handling(copy_flag):
    graph = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)

    lap, d = _laplacian_dense(
        graph, normed=False, axis=0, copy=copy_flag,
        form="array", dtype=None, symmetrized=False
    )

    assert lap.dtype == np.float32
    print(f"Test passed for copy={copy_flag}")

if __name__ == "__main__":
    # Run the hypothesis test
    test_laplacian_dense_dtype_handling()
    print("All tests passed!")