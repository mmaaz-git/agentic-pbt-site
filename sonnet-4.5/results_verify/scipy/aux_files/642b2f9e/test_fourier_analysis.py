import numpy as np
import scipy.fftpack as fftpack

# Test with simple array
x = np.array([1., 2., 3., 4.])

# Get the FFT of original
fft_original = np.fft.fft(x)
print("Original FFT:")
for i, val in enumerate(fft_original):
    print(f"  j={i}: {val}")

# Apply hilbert transform
h = fftpack.hilbert(x)
fft_hilbert = np.fft.fft(h)
print("\nFFT after hilbert:")
for i, val in enumerate(fft_hilbert):
    print(f"  j={i}: {val}")

# Apply ihilbert transform
ih = fftpack.ihilbert(h)
fft_ihilbert = np.fft.fft(ih)
print("\nFFT after ihilbert(hilbert(x)):")
for i, val in enumerate(fft_ihilbert):
    print(f"  j={i}: {val}")

print(f"\nOriginal: {x}")
print(f"After hilbert: {h}")
print(f"After ihilbert(hilbert(x)): {ih}")
print(f"\nDifference from original: {ih - x}")