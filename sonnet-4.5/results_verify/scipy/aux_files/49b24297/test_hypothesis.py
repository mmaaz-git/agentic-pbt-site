from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

@st.composite
def sorted_times_strategy(draw, min_times=2, max_times=5):
    n = draw(st.integers(min_value=min_times, max_value=max_times))
    times = sorted(draw(st.lists(
        st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n)))
    times = np.array(times)
    assume(len(np.unique(times)) == len(times))
    return times

@given(sorted_times_strategy())
@settings(max_examples=200)
def test_rotation_spline_boundary_conditions(times):
    """Property: RotationSpline should handle valid time arrays."""
    n = len(times)
    rotations = Rotation.random(n)
    try:
        spline = RotationSpline(times, rotations)
        print(f"Success with times={times}")
    except Exception as e:
        print(f"Failed with times={times}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_rotation_spline_boundary_conditions()