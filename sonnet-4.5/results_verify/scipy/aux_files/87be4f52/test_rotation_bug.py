#!/usr/bin/env python3
"""Test script to reproduce the scipy Rotation.mean() segmentation fault bug."""

import sys
import traceback

def test_basic_reproduction():
    """Test the basic bug reproduction case."""
    print("Testing basic reproduction case...")

    try:
        from scipy.spatial.transform import Rotation
        import numpy as np

        print("Creating quaternion: [0.0, 0.0, 0.0, 1.0]")
        q = np.array([0.0, 0.0, 0.0, 1.0])

        print("Creating Rotation object from quaternion...")
        r = Rotation.from_quat(q)
        print(f"Rotation object created: {r}")

        print("\nAttempting to call Rotation.mean([r])...")
        r_mean = Rotation.mean([r])

        print(f"Success! Mean rotation: {r_mean}")
        print(f"Mean quaternion: {r_mean.as_quat()}")

        return True

    except Exception as e:
        print(f"\nException occurred (not a segfault): {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_multiple_rotations():
    """Test that mean works with multiple rotations."""
    print("\n" + "="*60)
    print("Testing with multiple rotations...")

    try:
        from scipy.spatial.transform import Rotation
        import numpy as np

        print("Creating two rotations...")
        r1 = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        r2 = Rotation.from_quat([0.707, 0.0, 0.0, 0.707])  # 90 degree rotation around x

        print("Attempting to call Rotation.mean([r1, r2])...")
        r_mean = Rotation.mean([r1, r2])

        print(f"Success! Mean of two rotations: {r_mean}")
        print(f"Mean quaternion: {r_mean.as_quat()}")

        return True

    except Exception as e:
        print(f"\nException occurred: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_single_rotation_not_in_list():
    """Test passing a single rotation not wrapped in a list."""
    print("\n" + "="*60)
    print("Testing with single rotation not in a list...")

    try:
        from scipy.spatial.transform import Rotation
        import numpy as np

        print("Creating rotation...")
        r = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])

        print("Attempting to call Rotation.mean(r) - single rotation, not in list...")
        r_mean = Rotation.mean(r)

        print(f"Success! Mean: {r_mean}")
        print(f"Mean quaternion: {r_mean.as_quat()}")

        return True

    except Exception as e:
        print(f"\nException occurred: {type(e).__name__}: {e}")
        return False

def test_empty_list():
    """Test with empty list."""
    print("\n" + "="*60)
    print("Testing with empty list...")

    try:
        from scipy.spatial.transform import Rotation

        print("Attempting to call Rotation.mean([])...")
        r_mean = Rotation.mean([])

        print(f"Success! Mean of empty list: {r_mean}")

        return True

    except Exception as e:
        print(f"\nException occurred: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("SCIPY ROTATION.MEAN() BUG REPRODUCTION TEST")
    print("="*60)

    # Test the main bug case
    success1 = test_basic_reproduction()

    # Test other cases
    success2 = test_multiple_rotations()
    success3 = test_single_rotation_not_in_list()
    success4 = test_empty_list()

    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"- Single rotation in list: {'PASSED' if success1 else 'FAILED/SEGFAULT'}")
    print(f"- Multiple rotations: {'PASSED' if success2 else 'FAILED'}")
    print(f"- Single rotation (no list): {'PASSED' if success3 else 'FAILED'}")
    print(f"- Empty list: {'PASSED' if success4 else 'FAILED'}")

    if not success1:
        print("\n*** BUG CONFIRMED: Single rotation in list causes issue ***")
        print("If this script crashed with segfault, the bug is confirmed.")