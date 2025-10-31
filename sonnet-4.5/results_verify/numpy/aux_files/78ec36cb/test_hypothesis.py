import numpy as np
import numpy.linalg as LA
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra import numpy as npst

def matrices(min_side=2, max_side=5, dtype=np.float64):
    return st.integers(min_value=min_side, max_value=max_side).flatmap(
        lambda n: npst.arrays(
            dtype=dtype,
            shape=(n, n),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )

@given(matrices(min_side=2, max_side=4))
@settings(max_examples=50)
def test_eig_property(A):
    try:
        # Note: numpy.linalg.eig returns a tuple, not a namedtuple with attributes
        w, v = LA.eig(A)
        eigenvalues = w
        eigenvectors = v

        for i in range(len(eigenvalues)):
            v_i = eigenvectors[:, i]
            lam = eigenvalues[i]
            Av = A @ v_i
            lam_v = lam * v_i

            assert np.allclose(Av, lam_v, rtol=1e-4, atol=1e-6), \
                f"A @ v != λ * v for eigenvalue {i}. Max diff: {np.max(np.abs(Av - lam_v))}"
    except LA.LinAlgError:
        assume(False)

# Test the specific failing case
print("Testing specific failing case from bug report...")
A = np.array([[0.00000000e+000, 1.52474291e-300],
              [1.00000000e+000, 1.00000000e+000]])

try:
    w, v = LA.eig(A)
    eigenvalues = w
    eigenvectors = v

    for i in range(len(eigenvalues)):
        v_i = eigenvectors[:, i]
        lam = eigenvalues[i]
        Av = A @ v_i
        lam_v = lam * v_i

        assert np.allclose(Av, lam_v, rtol=1e-4, atol=1e-6), \
            f"A @ v != λ * v for eigenvalue {i}. Max diff: {np.max(np.abs(Av - lam_v))}"
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed as expected: {e}")

# Run hypothesis test
print("\nRunning hypothesis test...")
try:
    test_eig_property()
    print("All hypothesis tests passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")