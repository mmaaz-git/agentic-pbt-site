import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline


def test_rotation_spline_keyframe_exact(n_keyframes, seed):
    np.random.seed(seed)

    times = np.sort(np.random.rand(n_keyframes)) * 10
    times[0] = 0.0

    quats = np.random.randn(n_keyframes, 4)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    rotations = Rotation.from_quat(quats)

    spline = RotationSpline(times, rotations)

    for i, t in enumerate(times):
        interp_rot = spline(t)
        expected_quat = rotations[i].as_quat()
        actual_quat = interp_rot.as_quat()

        assert np.allclose(expected_quat, actual_quat, atol=1e-6) or np.allclose(expected_quat, -actual_quat, atol=1e-6), \
            f"RotationSpline at keyframe time {t} (index {i}) should return exact keyframe rotation.\nExpected: {expected_quat}\nActual: {actual_quat}"

# Test with specific failing inputs
def test_specific_cases():
    print("Testing n_keyframes=5, seed=493")
    try:
        test_rotation_spline_keyframe_exact(5, 493)
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")

    print("\nTesting n_keyframes=4, seed=4")
    try:
        test_rotation_spline_keyframe_exact(4, 4)
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")

if __name__ == "__main__":
    test_specific_cases()