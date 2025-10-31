from hypothesis import given, strategies as st, settings, assume
from scipy.spatial.transform import Rotation
import numpy as np
import hypothesis.extra.numpy as hnp

@st.composite
def quaternions(draw):
    q = draw(hnp.arrays(np.float64, (4,), elements=st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False)))
    assume(np.linalg.norm(q) > 1e-10)
    return q / np.linalg.norm(q)

def rotation_equal(r1, r2, atol=1e-10):
    """Check if two rotations are equal within tolerance"""
    q1 = r1.as_quat()
    q2 = r2.as_quat()
    # Account for quaternion double cover (q and -q represent same rotation)
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -q2, atol=atol)

@given(quaternions())
@settings(max_examples=200)
def test_rotation_mean_single(q):
    """Property: mean of single rotation should be itself"""
    r = Rotation.from_quat(q)
    r_mean = Rotation.mean([r])

    assert rotation_equal(r, r_mean, atol=1e-10), "Mean of single rotation is not itself"

if __name__ == "__main__":
    test_rotation_mean_single()