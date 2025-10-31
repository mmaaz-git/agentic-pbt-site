import numpy as np
from scipy.spatial.transform import Rotation

print("Analyzing the pattern of round-trip failures\n")

# Test systematic variations
failures = []
successes = []

# Generate many test cases
np.random.seed(123)
for i in range(200):
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
        failures.append((rv, rv1, rv2))
    else:
        successes.append((rv, rv1, rv2))

print(f"Results: {len(failures)} failures, {len(successes)} successes out of 200")
print()

# Analyze the difference between failures and successes
if failures:
    print("Analyzing failure patterns:")
    print("First 5 failures:")
    for i, (orig, r1, r2) in enumerate(failures[:5]):
        print(f"\n  Failure {i+1}:")
        print(f"    Original: {orig}")
        print(f"    Round 1:  {r1}")
        print(f"    Round 2:  {r2}")
        print(f"    r1 == -r2: {np.allclose(r1, -r2)}")

        # Check quaternion representation
        rot = Rotation.from_rotvec(orig)
        q = rot.as_quat()
        print(f"    Quaternion: {q}")
        print(f"    Quat scalar (w): {q[3]}")

print("\nFirst 5 successes:")
for i, (orig, r1, r2) in enumerate(successes[:5]):
    print(f"\n  Success {i+1}:")
    print(f"    Original: {orig}")
    print(f"    Round 1:  {r1}")

    # Check quaternion representation
    rot = Rotation.from_rotvec(orig)
    q = rot.as_quat()
    print(f"    Quaternion: {q}")
    print(f"    Quat scalar (w): {q[3]}")

# Check if there's a pattern in the quaternion scalar component
print("\n\nAnalyzing quaternion scalar components:")
failure_w = []
success_w = []

for orig, r1, r2 in failures:
    rot = Rotation.from_rotvec(r1)
    q = rot.as_quat()
    failure_w.append(abs(q[3]))

for orig, r1, r2 in successes:
    rot = Rotation.from_rotvec(r1)
    q = rot.as_quat()
    success_w.append(abs(q[3]))

if failure_w:
    print(f"Failure quaternion |w| range: [{min(failure_w):.10f}, {max(failure_w):.10f}]")
if success_w:
    print(f"Success quaternion |w| range: [{min(success_w):.10f}, {max(success_w):.10f}]")

# Check if numerical precision plays a role
print("\n\nChecking if it's a numerical precision issue:")
for i, (orig, r1, r2) in enumerate(failures[:3]):
    print(f"\nFailure {i+1}:")
    print(f"  |r1| = {np.linalg.norm(r1)}")
    print(f"  |r2| = {np.linalg.norm(r2)}")
    print(f"  |r1| - π = {np.linalg.norm(r1) - np.pi}")
    print(f"  |r2| - π = {np.linalg.norm(r2) - np.pi}")