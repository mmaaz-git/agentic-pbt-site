#!/usr/bin/env python3
"""
Property-based testing for RotationSpline bug using Hypothesis

This test demonstrates the RotationSpline bug where the spline doesn't always
pass through its control points when using specific non-uniform time spacing.

The bug manifests in several ways:
1. Control point mismatch: Spline doesn't evaluate to the expected rotation at control points
2. NaN/Inf values: Overflow in internal calculations leads to invalid results
3. Zero norm quaternions: Invalid rotations are produced during evaluation
"""

from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline
import warnings

# Strategy for generating sorted time arrays with potential problematic spacing
@st.composite
def sorted_times_strategy(draw, min_times=2, max_times=5):
    """Generate sorted time arrays that can trigger the RotationSpline bug."""
    n = draw(st.integers(min_value=min_times, max_value=max_times))
    times = sorted(draw(st.lists(
        st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n)))
    times = np.array(times)
    # Ensure all times are unique
    assume(len(np.unique(times)) == len(times))
    return times

def rotations_equal(r1, r2, atol=1e-5):
    """Check if two rotations are equivalent (accounting for quaternion double cover)."""
    q1 = r1.as_quat()
    q2 = r2.as_quat()
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -q2, atol=atol)

@given(sorted_times_strategy())
@settings(max_examples=100)
def test_rotation_spline_boundary_conditions(times):
    """
    Property: RotationSpline should exactly match control points.

    This is the fundamental interpolation property - a spline must pass through
    all its control points. This test has found failures with specific non-uniform
    time spacings.
    """
    n = len(times)
    rotations = Rotation.random(n)

    try:
        spline = RotationSpline(times, rotations)
    except ValueError as e:
        if "array must not contain infs or NaNs" in str(e):
            # This is one manifestation of the bug - overflow during construction
            print(f"OVERFLOW BUG: Construction failed with times={times}")
            raise AssertionError(f"RotationSpline construction failed due to overflow: {e}")
        else:
            raise

    # Test that the spline passes through each control point
    for i, t in enumerate(times):
        try:
            result = spline([t])
        except ValueError as e:
            if "Found zero norm quaternions" in str(e):
                # Another manifestation - zero norm quaternions during evaluation
                print(f"ZERO NORM BUG: Evaluation failed at t={t} with times={times}")
                raise AssertionError(f"RotationSpline evaluation produced zero norm quaternion: {e}")
            else:
                raise

        if not rotations_equal(result, rotations[i], atol=1e-5):
            # The core bug - spline doesn't match control points
            expected_quat = rotations[i].as_quat()
            result_quat = result.as_quat()
            diff1 = np.linalg.norm(expected_quat - result_quat)
            diff2 = np.linalg.norm(expected_quat + result_quat)
            min_diff = min(diff1, diff2)

            print(f"CONTROL POINT MISMATCH:")
            print(f"  Times: {times}")
            print(f"  Failed at control point {i}, t={t}")
            print(f"  Expected quaternion: {expected_quat}")
            print(f"  Got quaternion:      {result_quat}")
            print(f"  Minimum difference:  {min_diff}")

            raise AssertionError(f"RotationSpline doesn't match control point {i} at t={t}")

@given(sorted_times_strategy())
@settings(max_examples=50)
def test_rotation_spline_no_nans_infs(times):
    """
    Property: RotationSpline should never produce NaN or Inf values.

    This tests another aspect of the bug where numerical overflow leads to
    invalid calculations during spline construction or evaluation.
    """
    n = len(times)
    rotations = Rotation.random(n)

    # Suppress overflow warnings to catch them as exceptions instead
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)

        try:
            spline = RotationSpline(times, rotations)

            # Test evaluation at control points
            for t in times:
                result = spline([t])
                quat = result.as_quat()
                assert np.all(np.isfinite(quat)), f"Non-finite quaternion at t={t}: {quat}"

        except (RuntimeWarning, ValueError) as e:
            if any(keyword in str(e).lower() for keyword in ['overflow', 'nan', 'inf']):
                print(f"NUMERICAL INSTABILITY BUG:")
                print(f"  Times: {times}")
                print(f"  Error: {e}")
                raise AssertionError(f"RotationSpline produced numerical instability: {e}")
            else:
                raise

def reproduce_known_failure():
    """Reproduce a known failing case to demonstrate the bug."""
    print("=== Reproducing Known Failure ===")

    # This specific time configuration is known to cause failures
    times = np.array([0., 0.015625, 1., 2.])

    print(f"Testing with times: {times}")
    print("Searching for a seed that triggers the bug...")

    for seed in range(100):
        np.random.seed(seed)
        rotations = Rotation.random(4)

        try:
            spline = RotationSpline(times, rotations)

            # Check if spline passes through control points
            failed = False
            for i, t in enumerate(times):
                result = spline([t])
                expected_quat = rotations[i].as_quat()
                result_quat = result.as_quat()

                diff1 = np.linalg.norm(expected_quat - result_quat)
                diff2 = np.linalg.norm(expected_quat + result_quat)
                min_diff = min(diff1, diff2)

                if min_diff > 1e-5:
                    print(f"  ✗ FOUND BUG with seed {seed}")
                    print(f"    Mismatch at control point {i}, t={t}")
                    print(f"    Expected: {expected_quat}")
                    print(f"    Got:      {result_quat}")
                    print(f"    Difference: {min_diff}")
                    return True

        except (ValueError, RuntimeWarning) as e:
            print(f"  ✗ FOUND BUG with seed {seed}: {type(e).__name__}: {e}")
            return True

    print("  No bug found in 100 seeds (this is rare)")
    return False

if __name__ == "__main__":
    print("Property-Based Testing of RotationSpline Bug")
    print("=" * 50)

    # First, try to reproduce a known failure
    reproduce_known_failure()

    print("\n=== Running Property-Based Tests ===")
    print("Note: These tests are expected to fail, demonstrating the bug")

    try:
        print("\nTesting boundary conditions property...")
        test_rotation_spline_boundary_conditions()
        print("✓ All boundary condition tests passed")
    except Exception as e:
        print(f"✗ Boundary condition test failed: {e}")

    try:
        print("\nTesting numerical stability property...")
        test_rotation_spline_no_nans_infs()
        print("✓ All numerical stability tests passed")
    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}")

    print("\n=== Summary ===")
    print("The RotationSpline bug manifests in three ways:")
    print("1. Control point mismatch: Spline doesn't pass through control points")
    print("2. Numerical overflow: NaN/Inf values in calculations")
    print("3. Invalid rotations: Zero norm quaternions produced")
    print("\nThe bug is triggered by specific non-uniform time spacing patterns.")