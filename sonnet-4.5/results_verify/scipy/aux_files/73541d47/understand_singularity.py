import numpy as np
from scipy.spatial.transform import Rotation

print("Understanding the π rotation singularity\n")

# At π rotation, the quaternion has w=0, making the sign ambiguous
# Because q and -q represent the same rotation

# Let's verify the hypothesis test actually catches this
print("Running the Hypothesis test to find a failure...")

from hypothesis import given, settings, strategies as st
from hypothesis import Phase

@st.composite
def rotation_vectors(draw):
    axis = draw(st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
                         min_size=3, max_size=3))
    angle = draw(st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False))
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return [0, 0, 0]
    return (np.array(axis) / axis_norm * angle).tolist()

# Try to find failures
found_failure = False
failure_example = None

@given(st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
               min_size=3, max_size=3))
@settings(max_examples=1000, phases=[Phase.generate])
def test_find_failure(axis):
    global found_failure, failure_example

    if found_failure:
        return

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return

    # Create rotation vector with magnitude exactly π
    rv = (np.array(axis) / axis_norm * np.pi)

    r1 = Rotation.from_rotvec(rv)
    rv1 = r1.as_rotvec()
    r2 = Rotation.from_rotvec(rv1)
    rv2 = r2.as_rotvec()

    if not np.allclose(rv1, rv2, atol=1e-10):
        found_failure = True
        failure_example = (rv, rv1, rv2, r1, r2)
        print(f"Found a failure!")
        print(f"  Input:   {rv}")
        print(f"  Round 1: {rv1}")
        print(f"  Round 2: {rv2}")

# Try to find a failure
test_find_failure()

if failure_example:
    rv, rv1, rv2, r1, r2 = failure_example
    print("\nDetailed analysis of the failure:")
    print(f"  Original magnitude: {np.linalg.norm(rv)}")
    print(f"  Round 1 magnitude:  {np.linalg.norm(rv1)}")
    print(f"  Round 2 magnitude:  {np.linalg.norm(rv2)}")
    print(f"  r1 quaternion: {r1.as_quat()}")
    print(f"  r2 quaternion: {r2.as_quat()}")
    print(f"  r1 matrix == r2 matrix: {np.allclose(r1.as_matrix(), r2.as_matrix())}")

# Let's test the mathematical theory:
print("\n\nMathematical analysis:")
print("For rotations of exactly π radians:")
print("- The quaternion scalar component w = cos(θ/2) = cos(π/2) = 0")
print("- This creates a singularity where q = [x,y,z,0] and -q = [-x,-y,-z,0]")
print("- Both represent the same rotation, but converting back gives different rotvecs")

# Create a specific test case
print("\n\nControlled test with exact π rotation:")
axis = np.array([1, 0, 0])  # x-axis
angle = np.pi
rv_test = axis * angle

print(f"Input rotation vector: {rv_test} (π rotation around x-axis)")

r = Rotation.from_rotvec(rv_test)
q = r.as_quat()
print(f"Quaternion: {q}")
print(f"Quaternion w component: {q[3]} (should be ~0)")

# Now test with a small perturbation away from π
print("\n\nTest with angle slightly less than π:")
angle = np.pi - 1e-10
rv_test = axis * angle
r = Rotation.from_rotvec(rv_test)
rv1 = r.as_rotvec()
r2 = Rotation.from_rotvec(rv1)
rv2 = r2.as_rotvec()
print(f"Angle = π - 1e-10")
print(f"Round-trip success: {np.allclose(rv1, rv2, atol=1e-12)}")

print("\n\nTest with angle slightly more than π (will wrap):")
angle = np.pi + 1e-10
rv_test = axis * angle
r = Rotation.from_rotvec(rv_test)
rv1 = r.as_rotvec()
print(f"Input magnitude: {np.linalg.norm(rv_test)}")
print(f"Output magnitude: {np.linalg.norm(rv1)}")
print(f"Note: angles > π wrap around to the [0, π] range")