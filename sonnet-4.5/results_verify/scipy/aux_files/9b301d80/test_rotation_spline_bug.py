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
def test_rotation_spline_produces_valid_rotations(times):
    """Property: RotationSpline should produce valid rotations at any time."""
    n = len(times)
    rotations = Rotation.random(n)
    spline = RotationSpline(times, rotations)

    test_times = []
    for i in range(len(times) - 1):
        test_times.append((times[i] + times[i+1]) / 2)

    if test_times:
        try:
            results = spline(test_times)
            print(f"SUCCESS: times={times}")
        except ValueError as e:
            print(f"FAILURE: times={times}")
            print(f"Error: {e}")
            raise

if __name__ == "__main__":
    test_rotation_spline_produces_valid_rotations()