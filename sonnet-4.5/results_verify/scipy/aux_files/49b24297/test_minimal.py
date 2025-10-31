import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline
import traceback

# Test the specific failing case from the bug report
times = np.array([0., 0.0078125, 1., 5.])
rotations = Rotation.from_quat([
    [0.5, 0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5, 0.5]
])

print("Testing with times:", times)
print("Times are strictly increasing:", np.all(np.diff(times) > 0))
print("Rotations are valid (unit quaternions):", np.allclose(np.linalg.norm(rotations.as_quat(), axis=1), 1))
print("Time delta ratios:", np.max(np.diff(times)) / np.min(np.diff(times)))

try:
    spline = RotationSpline(times, rotations)
    print("SUCCESS: RotationSpline created successfully")
except ValueError as e:
    print(f"FAILED with ValueError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"FAILED with {type(e).__name__}: {e}")
    traceback.print_exc()