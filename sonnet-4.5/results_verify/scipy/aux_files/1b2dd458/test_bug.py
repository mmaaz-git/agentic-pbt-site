#!/usr/bin/env python3
"""Test the grayscale conversion coefficients bug"""

import inspect
import scipy.datasets
import re

print("=" * 60)
print("Testing scipy.datasets.face Grayscale Conversion Bug")
print("=" * 60)

# Test 1: Run the property-based test from the bug report
def test_grayscale_coefficients_sum():
    """Grayscale coefficients should sum to 1.0 for brightness preservation"""
    source = inspect.getsource(scipy.datasets.face)

    # Extract coefficients: 0.21, 0.71, 0.07
    pattern = r'(\d+\.\d+)\s*\*\s*face\[:,\s*:,\s*0\]\s*\+\s*(\d+\.\d+)\s*\*\s*face\[:,\s*:,\s*1\]\s*\+\s*(\d+\.\d+)\s*\*\s*face\[:,\s*:,\s*2\]'
    match = re.search(pattern, source)

    if match:
        r_coeff, g_coeff, b_coeff = float(match.group(1)), float(match.group(2)), float(match.group(3))
        total = r_coeff + g_coeff + b_coeff

        print(f"\nExtracted coefficients from source code:")
        print(f"  Red coefficient:   {r_coeff}")
        print(f"  Green coefficient: {g_coeff}")
        print(f"  Blue coefficient:  {b_coeff}")
        print(f"  Sum of coefficients: {total}")

        try:
            assert abs(total - 1.0) < 0.001, \
                f"Coefficients ({r_coeff}, {g_coeff}, {b_coeff}) sum to {total}, expected 1.0"
            print("  ✓ Test PASSED: Coefficients sum to 1.0")
        except AssertionError as e:
            print(f"  ✗ Test FAILED: {e}")
            return False
    else:
        print("  Could not extract coefficients from source")
        return None

    return True

# Test 2: Direct verification of coefficients
print("\n1. Direct verification of coefficients:")
coefficients = [0.21, 0.71, 0.07]
total = sum(coefficients)
print(f"  Coefficients: {coefficients}")
print(f"  Sum: {total}")
print(f"  Difference from 1.0: {1.0 - total}")

# Test 3: Check against common grayscale conversion standards
print("\n2. Common grayscale conversion standards:")
print("  Rec. 601 (NTSC):  0.299R + 0.587G + 0.114B = 1.0")
print("  Rec. 709 (HDTV):  0.2126R + 0.7152G + 0.0722B = 1.0")
print("  SciPy implementation: 0.21R + 0.71G + 0.07B = 0.99")

# Test 4: Calculate brightness loss
print("\n3. Brightness loss calculation:")
brightness_loss = (1.0 - total) * 100
print(f"  Brightness loss: {brightness_loss:.1f}%")

# Test 5: Run the property-based test
print("\n4. Running property-based test:")
result = test_grayscale_coefficients_sum()

# Test 6: Test actual grayscale conversion
print("\n5. Testing actual grayscale conversion:")
try:
    import numpy as np
    # Create a test image with known values
    test_img = np.ones((2, 2, 3), dtype=np.uint8) * 100

    # Manual calculation with current coefficients
    manual_gray = (0.21 * test_img[:, :, 0] +
                   0.71 * test_img[:, :, 1] +
                   0.07 * test_img[:, :, 2])

    print(f"  Input RGB value: [100, 100, 100]")
    print(f"  Expected grayscale (if sum=1.0): 100")
    print(f"  Actual grayscale (sum=0.99): {manual_gray[0,0]:.2f}")
    print(f"  Difference: {100 - manual_gray[0,0]:.2f}")

except Exception as e:
    print(f"  Error in conversion test: {e}")

# Test 7: Check what happens with the proposed fix
print("\n6. Testing proposed fix (coefficients = [0.21, 0.71, 0.08]):")
fixed_coefficients = [0.21, 0.71, 0.08]
fixed_total = sum(fixed_coefficients)
print(f"  Fixed coefficients: {fixed_coefficients}")
print(f"  Sum: {fixed_total}")
print(f"  Difference from 1.0: {abs(1.0 - fixed_total)}")

print("\n" + "=" * 60)
print("Summary:")
print("  - Current coefficients sum to 0.99 (not 1.0)")
print("  - This causes 1% brightness loss in grayscale conversion")
print("  - The proposed fix would make coefficients sum to 1.0")
print("=" * 60)