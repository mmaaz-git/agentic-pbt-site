from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

@given(st.integers(min_value=1, max_value=5), st.random_module())
@settings(max_examples=50)
def test_expm_return_type_matches_documentation(n, random):
    A_dense = np.random.rand(n, n) * 0.1
    A = sp.csc_array(A_dense)

    result = sla.expm(A)

    assert isinstance(result, np.ndarray), \
        f"Documentation says expm returns ndarray, but got {type(result)}"

# Run the test
if __name__ == "__main__":
    test_expm_return_type_matches_documentation()
    print("Test passed!")