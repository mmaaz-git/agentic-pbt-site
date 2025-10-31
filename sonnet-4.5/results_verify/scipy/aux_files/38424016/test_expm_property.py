import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from hypothesis import given, strategies as st, settings


@given(st.integers(min_value=2, max_value=6))
@settings(max_examples=50)
def test_expm_return_type_documentation(n):
    dense_A = np.random.randn(n, n)
    sparse_A = sp.csr_array(dense_A)

    result_dense = spl.expm(dense_A)
    result_sparse = spl.expm(sparse_A)

    assert isinstance(result_dense, np.ndarray), \
        "expm with dense input should return ndarray"

    is_sparse_output = sp.issparse(result_sparse)
    is_ndarray_output = isinstance(result_sparse, np.ndarray)

    assert is_sparse_output or is_ndarray_output, \
        f"expm should return either sparse or ndarray, got {type(result_sparse)}"

    print(f"Test n={n}:")
    print(f"  Dense result type: {type(result_dense)}")
    print(f"  Sparse result type: {type(result_sparse)}")
    print(f"  Is sparse output sparse?: {is_sparse_output}")
    print(f"  Is sparse output ndarray?: {is_ndarray_output}")

if __name__ == "__main__":
    test_expm_return_type_documentation()