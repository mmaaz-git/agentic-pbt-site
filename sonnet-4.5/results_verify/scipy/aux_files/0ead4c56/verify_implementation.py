#!/usr/bin/env python3
"""Verify how convolve1d actually implements the reversal."""

import numpy as np
import scipy.ndimage as ndimage

def manual_correlate_constant(input_arr, weights, origin=0):
    """Manually compute correlation with constant boundary mode."""
    n = len(input_arr)
    w_len = len(weights)
    result = np.zeros(n)

    # Origin shifts where the center of the kernel is
    center = (w_len - 1) // 2

    for i in range(n):
        sum_val = 0.0
        for j in range(w_len):
            # Position in input array
            pos = i + j - center - origin
            if 0 <= pos < n:
                sum_val += input_arr[pos] * weights[j]
            # else: implicitly multiply by 0 (constant mode with cval=0)
        result[i] = sum_val

    return result

def test_manual_implementation():
    """Test our manual implementation against scipy."""
    print("Testing manual correlation implementation")
    print("=" * 60)

    input_arr = np.array([1., 2., 3., 4., 5.])
    weights = np.array([1., 2., 3.])

    print(f"Input: {input_arr}")
    print(f"Weights: {weights}")

    for origin in [-1, 0, 1]:
        manual = manual_correlate_constant(input_arr, weights, origin)
        scipy_result = ndimage.correlate1d(input_arr, weights, mode='constant', origin=origin, cval=0.0)

        print(f"\nOrigin={origin}:")
        print(f"Manual:  {manual}")
        print(f"SciPy:   {scipy_result}")
        print(f"Match: {np.allclose(manual, scipy_result)}")

def test_convolution_is_reversed_correlation():
    """Verify that convolution reverses the weights internally."""
    print("\n" + "=" * 60)
    print("Testing if convolve1d reverses weights internally")
    print("=" * 60)

    input_arr = np.array([1., 2., 3., 4., 5.])
    weights = np.array([1., 2., 3.])

    # Convolution should be correlation with reversed weights
    conv = ndimage.convolve1d(input_arr, weights, mode='constant', origin=0, cval=0.0)
    corr_reversed = ndimage.correlate1d(input_arr, weights[::-1], mode='constant', origin=0, cval=0.0)

    print(f"convolve1d(weights):           {conv}")
    print(f"correlate1d(reversed weights): {corr_reversed}")
    print(f"Equal? {np.allclose(conv, corr_reversed)}")

    # But the bug report says the opposite should also be true
    corr = ndimage.correlate1d(input_arr, weights, mode='constant', origin=0, cval=0.0)
    conv_reversed = ndimage.convolve1d(input_arr, weights[::-1], mode='constant', origin=0, cval=0.0)

    print(f"\ncorrelate1d(weights):          {corr}")
    print(f"convolve1d(reversed weights):  {conv_reversed}")
    print(f"Equal? {np.allclose(corr, conv_reversed)}")

if __name__ == "__main__":
    test_manual_implementation()
    test_convolution_is_reversed_correlation()