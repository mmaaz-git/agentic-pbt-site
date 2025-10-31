import numpy as np
import numpy.linalg as la
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

@given(arrays(dtype=np.float64, shape=(3, 3), elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
@settings(max_examples=300)
def test_eig_property(a):
    try:
        result = la.eig(a)
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors
    except la.LinAlgError:
        return

    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]

        lhs = a @ v
        rhs = lam * v

        if np.linalg.norm(v) > 1e-10:
            assert np.allclose(lhs, rhs, rtol=1e-4, atol=1e-7), f"A @ v != lambda * v for eigenpair {i}"

# Run the test to find a failure
if __name__ == "__main__":
    test_eig_property()
    print("All tests passed!")