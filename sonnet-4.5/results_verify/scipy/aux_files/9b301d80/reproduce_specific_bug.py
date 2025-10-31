import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

# Test the specific failing example from the bug report
times = np.array([0., 0.0078125, 1., 4.])
np.random.seed(43)
rotations = Rotation.random(4)

print(f"Times: {times}")
print(f"Number of rotations: {len(rotations)}")

try:
    spline = RotationSpline(times, rotations)
    print("Spline created successfully")

    t_mid = 0.5
    print(f"Evaluating spline at t={t_mid}")
    result = spline([t_mid])
    print(f"Result: {result}")
    print("SUCCESS - No error occurred")
except ValueError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")