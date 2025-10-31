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
@settings(max_examples=1000)
def test_singular_matrix_has_reduced_rank(A):
    assume(np.all(np.isfinite(A)))
    det = np.linalg.det(A)
    rank = np.linalg.matrix_rank(A)

    if det == 0:
        assert rank < A.shape[0], f"Singular matrix (det=0) must have rank < {A.shape[0]}, got rank={rank}"

# Test with the specific failing input
print("Testing with the specific failing input from the bug report:")
A = np.array([[0.00000000e+000, 2.22507386e-311],
              [2.22507386e-311, 2.22507386e-311]])

det = np.linalg.det(A)
rank = np.linalg.matrix_rank(A)

print(f"Matrix A:")
print(A)
print(f"\nDeterminant: {det}")
print(f"Determinant == 0: {det == 0}")
print(f"matrix_rank: {rank}")
print(f"Expected rank: < 2 (since det=0)")

U, s, Vh = np.linalg.svd(A)
default_tol = s.max() * max(A.shape) * np.finfo(float).eps
print(f"\nSingular values: {s}")
print(f"Default tolerance: {default_tol}")
print(f"Number of singular values > default_tol: {np.sum(s > default_tol)}")

# Additional analysis
print(f"\nAdditional analysis:")
print(f"s.max(): {s.max()}")
print(f"max(A.shape): {max(A.shape)}")
print(f"np.finfo(float).eps: {np.finfo(float).eps}")
print(f"np.finfo(float).tiny: {np.finfo(float).tiny}")
print(f"Are all values subnormal? {np.all(np.abs(A.flatten()) < np.finfo(float).tiny)}")

# Calculate rank manually based on singular values
manual_rank = np.sum(s > default_tol)
print(f"\nManual rank calculation (s > default_tol): {manual_rank}")

# Calculate condition number
cond = np.linalg.cond(A)
print(f"Condition number: {cond}")

# Check if matrix is actually singular mathematically
print(f"\nMathematical analysis:")
print(f"First row: {A[0]}")
print(f"Second row: {A[1]}")

# Check if rows are linearly dependent
if A.shape[0] == 2:
    # For a 2x2 matrix, check if determinant is 0
    manual_det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    print(f"Manual determinant calculation: {manual_det}")
    print(f"Manual det == 0: {manual_det == 0}")

# Test with hypothesis
print("\n\nRunning hypothesis test (limited to 10 examples):")
try:
    @given(square_matrices(min_side=2, max_side=8))
    @settings(max_examples=10)
    def test_limited(A):
        assume(np.all(np.isfinite(A)))
        det = np.linalg.det(A)
        rank = np.linalg.matrix_rank(A)
        if det == 0:
            assert rank < A.shape[0], f"Singular matrix (det=0) must have rank < {A.shape[0]}, got rank={rank}, A={A}"

    test_limited()
    print("Hypothesis test passed for 10 examples")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")