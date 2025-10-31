from hypothesis import given, strategies as st
from scipy.spatial.transform import Rotation
import numpy as np

@given(st.integers(1, 100))
def test_reduce_identity_composition(seed):
    """Test that reducing a rotation composed with identity gives the original rotation"""
    np.random.seed(seed)
    r = Rotation.random()

    identity = Rotation.identity()
    composed = r * identity

    reduced = composed.reduce(left=identity)

    assert r.approx_equal(reduced, atol=1e-14)

# Run the test
if __name__ == "__main__":
    import traceback
    try:
        test_reduce_identity_composition(1)  # Hypothesis test accepts the value directly
        print("Test passed with seed=1")
    except Exception as e:
        print(f"Test failed with seed=1: {e}")
        traceback.print_exc()