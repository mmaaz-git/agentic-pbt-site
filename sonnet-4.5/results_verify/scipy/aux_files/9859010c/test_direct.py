from scipy.spatial.transform import Rotation
import numpy as np

# Direct test of the issue
def test_reduce_identity_composition_direct(seed):
    """Test that reducing a rotation composed with identity gives the original rotation"""
    np.random.seed(seed)
    r = Rotation.random()

    identity = Rotation.identity()
    composed = r * identity

    try:
        reduced = composed.reduce(left=identity)
        print(f"Test passed: reduced = {reduced}")
        assert r.approx_equal(reduced, atol=1e-14)
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test with seed=1
print("Testing with seed=1:")
test_reduce_identity_composition_direct(1)