import numpy as np
from hypothesis import given, assume, settings
from hypothesis.extra import numpy as npst
from hypothesis import strategies as st

def square_matrices(min_side=1, max_side=10):
    side = st.integers(min_value=min_side, max_value=max_side)
    return side.flatmap(
        lambda n: npst.arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        )
    )

@given(square_matrices(min_side=2, max_side=8))
@settings(max_examples=300)
def test_eig_reconstruction(A):
    assume(np.all(np.isfinite(A)))

    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        assume(np.all(np.isfinite(eigenvalues)) and np.all(np.isfinite(eigenvectors)))

        for i in range(len(eigenvalues)):
            Av = A @ eigenvectors[:, i]
            lambda_v = eigenvalues[i] * eigenvectors[:, i]
            assert np.allclose(Av, lambda_v, rtol=1e-6, atol=1e-9)
    except np.linalg.LinAlgError:
        assume(False)

if __name__ == "__main__":
    test_eig_reconstruction()