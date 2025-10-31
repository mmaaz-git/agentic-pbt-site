import numpy as np
import scipy.fftpack as fftpack

print("Testing even vs odd length arrays:")

# Even length arrays
even_arrays = [
    np.array([1., 2.]),  # len=2
    np.array([1., 2., 3., 4.]),  # len=4
    np.array([1., 2., 3., 4., 5., 6.]),  # len=6
]

# Odd length arrays
odd_arrays = [
    np.array([1., 2., 3.]),  # len=3
    np.array([1., 2., 3., 4., 5.]),  # len=5
    np.array([1., 2., 3., 4., 5., 6., 7.]),  # len=7
]

print("\nEVEN LENGTH ARRAYS:")
for x in even_arrays:
    result = fftpack.ihilbert(fftpack.hilbert(x))
    match = np.allclose(result, x, rtol=1e-10, atol=1e-12)
    print(f"len={len(x)}: {x} -> {result}, match={match}")

    # Check FFT to see what happens at Nyquist
    fft_orig = np.fft.fft(x)
    fft_after = np.fft.fft(result)
    nyquist_idx = len(x) // 2
    print(f"  DC component (j=0): orig={fft_orig[0]:.2f}, after={fft_after[0]:.2f}")
    print(f"  Nyquist (j={nyquist_idx}): orig={fft_orig[nyquist_idx]:.2f}, after={fft_after[nyquist_idx]:.2f}")

print("\nODD LENGTH ARRAYS:")
for x in odd_arrays:
    result = fftpack.ihilbert(fftpack.hilbert(x))
    match = np.allclose(result, x, rtol=1e-10, atol=1e-12)
    print(f"len={len(x)}: {x} -> {result}, match={match}")

    # Check FFT
    fft_orig = np.fft.fft(x)
    fft_after = np.fft.fft(result)
    print(f"  DC component (j=0): orig={fft_orig[0]:.2f}, after={fft_after[0]:.2f}")