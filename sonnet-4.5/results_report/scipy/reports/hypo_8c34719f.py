from hypothesis import given, strategies as st, settings
from scipy.spatial.transform import Rotation
import numpy as np

@st.composite
def rotation_strategy(draw):
    rotvec = draw(st.lists(
        st.floats(min_value=-2*np.pi, max_value=2*np.pi,
                  allow_nan=False, allow_infinity=False),
        min_size=3, max_size=3
    ))
    return Rotation.from_rotvec(rotvec)

@given(rotation_strategy())
@settings(max_examples=100)
def test_reduce_self_is_identity(r):
    """Reducing a rotation by a group containing itself should yield identity"""
    group = Rotation.concatenate([r])
    reduced = r.reduce(group)
    assert np.isclose(reduced.magnitude(), 0.0, atol=1e-10), \
        f"Expected magnitude 0.0, got {reduced.magnitude()}"

if __name__ == "__main__":
    test_reduce_self_is_identity()