#!/usr/bin/env python3
"""Test correct usage of Kleene operations with only one mask as None"""

import numpy as np
from pandas.core.ops import kleene_and, kleene_or, kleene_xor

def test_kleene_operations_correct_usage():
    """Test Kleene operations with correct usage (only one mask None)"""

    # Create an array and mask
    arr = np.array([True, False, True, False])
    mask = np.array([False, False, True, False])

    print("Testing with left as array, right as scalar (right_mask=None):")
    print(f"Array: {arr}")
    print(f"Mask: {mask}")
    print()

    # Test kleene_and with scalar right
    print("kleene_and(arr, True, mask, None):")
    result, result_mask = kleene_and(arr, True, mask, None)
    print(f"  Result: {result}")
    print(f"  Mask: {result_mask}")

    # Test kleene_or with scalar right
    print("\nkleene_or(arr, False, mask, None):")
    result, result_mask = kleene_or(arr, False, mask, None)
    print(f"  Result: {result}")
    print(f"  Mask: {result_mask}")

    # Test kleene_xor with scalar right
    print("\nkleene_xor(arr, True, mask, None):")
    result, result_mask = kleene_xor(arr, True, mask, None)
    print(f"  Result: {result}")
    print(f"  Mask: {result_mask}")

    print("\n" + "="*60)
    print("Testing with right as array, left as scalar (left_mask=None):")

    # Test kleene_and with scalar left
    print("\nkleene_and(False, arr, None, mask):")
    result, result_mask = kleene_and(False, arr, None, mask)
    print(f"  Result: {result}")
    print(f"  Mask: {result_mask}")

    # Test kleene_or with scalar left
    print("\nkleene_or(True, arr, None, mask):")
    result, result_mask = kleene_or(True, arr, None, mask)
    print(f"  Result: {result}")
    print(f"  Mask: {result_mask}")

    # Test kleene_xor with scalar left
    print("\nkleene_xor(False, arr, None, mask):")
    result, result_mask = kleene_xor(False, arr, None, mask)
    print(f"  Result: {result}")
    print(f"  Mask: {result_mask}")

    print("\n" + "="*60)
    print("All operations with correct usage completed successfully!")

if __name__ == "__main__":
    test_kleene_operations_correct_usage()