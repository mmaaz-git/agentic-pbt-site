from hypothesis import assume, given, settings, strategies as st
from hypothesis.extra import numpy as hpnp
import numpy as np


@given(
    hpnp.arrays(
        dtype=np.float64,
        shape=hpnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=5),
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
    ),
    hpnp.arrays(
        dtype=np.float64,
        shape=hpnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=5),
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=200)
def test_matrix_rank_product_bound(a, b):
    assume(a.shape[1] == b.shape[0])

    rank_a = np.linalg.matrix_rank(a)
    rank_b = np.linalg.matrix_rank(b)
    rank_ab = np.linalg.matrix_rank(a @ b)

    assert rank_ab <= min(rank_a, rank_b), f"rank(AB)={rank_ab} > min(rank(A)={rank_a}, rank(B)={rank_b})"

# Test with the specific failing input
print("Testing with the specific failing input...")
a = np.array([[5.e-324, 1.e+000],
              [5.e-324, 5.e-324]])
b = np.array([[2.e+000, 5.e-324],
              [5.e-324, 5.e-324]])

rank_a = np.linalg.matrix_rank(a)
rank_b = np.linalg.matrix_rank(b)
ab = a @ b
rank_ab = np.linalg.matrix_rank(ab)

print(f"Matrix A:\n{a}")
print(f"Matrix B:\n{b}")
print(f"Matrix AB:\n{ab}")
print(f"rank(A) = {rank_a}")
print(f"rank(B) = {rank_b}")
print(f"rank(AB) = {rank_ab}")
print(f"min(rank(A), rank(B)) = {min(rank_a, rank_b)}")
print(f"Is rank(AB) <= min(rank(A), rank(B))? {rank_ab <= min(rank_a, rank_b)}")

# Also run the hypothesis test
print("\nRunning hypothesis test...")
test_matrix_rank_product_bound()
print("Test completed")