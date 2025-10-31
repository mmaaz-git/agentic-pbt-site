import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

times = np.array([0.0, 6.64446271, 9.24290286, 9.3721529, 9.53565684])
quats = np.array([
    [0.70966851, 0.62720382, 0.3207292, 0.0108974],
    [0.46483691, 0.04322698, 0.86833592, -0.16748374],
    [-0.63456588, -0.47115133, 0.61052646, -0.05099029],
    [0.04451323, -0.38889422, -0.59025565, -0.70595901],
    [-0.29974423, -0.77272482, -0.38528129, -0.40571921]
])

rotations = Rotation.from_quat(quats)
spline = RotationSpline(times, rotations)

print("Testing all keyframes:")
for i, t in enumerate(times):
    expected = rotations[i].as_quat()
    actual = spline(t).as_quat()
    match = np.allclose(expected, actual, atol=1e-6) or np.allclose(expected, -actual, atol=1e-6)
    print(f"\nTime {t:.8f} (keyframe {i}):")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {actual}")
    print(f"  Match: {match}")

print("\n" + "="*60)
print("Specific bug case - last keyframe:")
expected = rotations[-1].as_quat()
actual = spline(times[-1]).as_quat()
print(f"Expected: {expected}")
print(f"Actual:   {actual}")
print(f"Match: {np.allclose(expected, actual, atol=1e-6) or np.allclose(expected, -actual, atol=1e-6)}")