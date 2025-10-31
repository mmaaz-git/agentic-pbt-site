#!/usr/bin/env python3
"""Hypothesis-based property test for scipy Rotation.mean() bug."""

from hypothesis import given, strategies as st, settings, assume
import hypothesis.extra.numpy as hnp
from scipy.spatial.transform import Rotation
import numpy as np

@st.composite
def quaternions(draw):
    """Generate normalized quaternions."""
    q = draw(hnp.arrays(np.float64, (4,),
                        elements=st.floats(min_value=-1, max_value=1,
                                         allow_nan=False, allow_infinity=False)))
    norm = np.linalg.norm(q)
    assume(norm > 1e-10)
    return q / norm

def rotation_equal(r1, r2, atol=1e-10):
    """Check if two rotations are approximately equal."""
    # Compare quaternions (accounting for the fact that q and -q represent the same rotation)
    q1 = r1.as_quat()
    q2 = r2.as_quat()

    # Check if either q1 ≈ q2 or q1 ≈ -q2
    return (np.allclose(q1, q2, atol=atol) or
            np.allclose(q1, -q2, atol=atol))

@given(quaternions())
@settings(max_examples=10)  # Reduced examples since we expect it to crash
def test_rotation_mean_single(q):
    """Property: mean of single rotation should be itself"""
    print(f"\nTesting with quaternion: {q}")

    r = Rotation.from_quat(q)
    print(f"Created rotation: {r}")

    print("Attempting Rotation.mean([r])...")
    r_mean = Rotation.mean([r])

    print(f"Success! Mean: {r_mean}")
    assert rotation_equal(r, r_mean, atol=1e-10), "Mean of single rotation is not itself"
    print("Property verified: mean equals original rotation")

if __name__ == "__main__":
    print("="*60)
    print("HYPOTHESIS PROPERTY TEST FOR ROTATION.MEAN()")
    print("="*60)

    print("\nFirst, let's test with the specific example from the bug report:")
    try:
        q = np.array([0.0, 0.0, 0.0, 1.0])
        print(f"Testing with q = {q}")
        r = Rotation.from_quat(q)
        print(f"Created rotation: {r}")

        print("Attempting Rotation.mean([r])...")
        r_mean = Rotation.mean([r])

        print(f"Success! Mean: {r_mean}")
        print(f"Mean quaternion: {r_mean.as_quat()}")

    except Exception as e:
        print(f"Exception: {e}")

    print("\n" + "="*60)
    print("Now running Hypothesis property tests...")
    print("="*60)

    try:
        test_rotation_mean_single()
        print("\nAll property tests passed!")
    except Exception as e:
        print(f"\nProperty test failed with exception: {e}")