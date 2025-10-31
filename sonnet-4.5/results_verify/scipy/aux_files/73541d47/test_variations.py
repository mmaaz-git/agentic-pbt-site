import numpy as np
from scipy.spatial.transform import Rotation

# Let's test different rotations around pi magnitude
print("Testing various rotations with magnitude near or at π\n")

# Test 1: Create rotation from rotation vector with magnitude exactly π
print("Test 1: Direct rotvec with magnitude π")
rv_pi = np.array([0, np.pi, 0])  # Rotation of π around y-axis
r1 = Rotation.from_rotvec(rv_pi)
rv1_out = r1.as_rotvec()
r2 = Rotation.from_rotvec(rv1_out)
rv2_out = r2.as_rotvec()
print(f"Input:  {rv_pi}")
print(f"Round1: {rv1_out}")
print(f"Round2: {rv2_out}")
print(f"Round1 == Round2: {np.allclose(rv1_out, rv2_out)}")
print()

# Test 2: Try with a different axis
print("Test 2: Different axis with magnitude π")
rv_pi2 = np.array([np.pi/np.sqrt(2), np.pi/np.sqrt(2), 0])  # magnitude = π
r1 = Rotation.from_rotvec(rv_pi2)
rv1_out = r1.as_rotvec()
r2 = Rotation.from_rotvec(rv1_out)
rv2_out = r2.as_rotvec()
print(f"Input:  {rv_pi2}")
print(f"|Input| = {np.linalg.norm(rv_pi2)}")
print(f"Round1: {rv1_out}")
print(f"Round2: {rv2_out}")
print(f"Round1 == Round2: {np.allclose(rv1_out, rv2_out)}")
print()

# Test 3: Create many random rotations with magnitude exactly π
print("Test 3: Random rotations with magnitude exactly π")
np.random.seed(42)
failures = 0
for i in range(100):
    # Random unit axis
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    # Scale to magnitude π
    rv = axis * np.pi

    # Test round-trip
    r1 = Rotation.from_rotvec(rv)
    rv1 = r1.as_rotvec()
    r2 = Rotation.from_rotvec(rv1)
    rv2 = r2.as_rotvec()

    if not np.allclose(rv1, rv2, atol=1e-10):
        failures += 1
        if failures == 1:  # Print first failure
            print(f"First failure:")
            print(f"  Original: {rv}")
            print(f"  Round1:   {rv1}")
            print(f"  Round2:   {rv2}")
            print(f"  Negated:  {np.allclose(rv1, -rv2)}")

print(f"Failed {failures}/100 round-trip tests with magnitude π")
print()

# Test 4: Test the exact matrix from the bug report
print("Test 4: Exact matrix from bug report")
matrix = np.array([[-1.,  0.,  0.],
                   [ 0.,  0.6, 0.8],
                   [ 0.,  0.8, -0.6]])

r = Rotation.from_matrix(matrix)
print(f"Magnitude: {r.magnitude()}")

# Multiple round trips
rv0 = r.as_rotvec()
print(f"Round 0: {rv0}")

r1 = Rotation.from_rotvec(rv0)
rv1 = r1.as_rotvec()
print(f"Round 1: {rv1}")

r2 = Rotation.from_rotvec(rv1)
rv2 = r2.as_rotvec()
print(f"Round 2: {rv2}")

r3 = Rotation.from_rotvec(rv2)
rv3 = r3.as_rotvec()
print(f"Round 3: {rv3}")

print(f"\nRound 0 == Round 1: {np.allclose(rv0, rv1)}")
print(f"Round 1 == Round 2: {np.allclose(rv1, rv2)}")
print(f"Round 2 == Round 3: {np.allclose(rv2, rv3)}")