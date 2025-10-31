import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

# Test a failing case found by Hypothesis
times = np.array([1., 1.0078125, 2., 4.])
np.random.seed(42)  # Fixed seed for reproducibility
rotations = Rotation.random(4)

print(f"Times: {times}")
print(f"Number of rotations: {len(rotations)}")

try:
    spline = RotationSpline(times, rotations)
    print("Spline created successfully")

    # Calculate mid-points for evaluation
    test_times = []
    for i in range(len(times) - 1):
        test_times.append((times[i] + times[i+1]) / 2)

    print(f"Evaluating spline at times: {test_times}")
    result = spline(test_times)
    print(f"Result: {result}")
    print("SUCCESS - No error occurred")
except ValueError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")