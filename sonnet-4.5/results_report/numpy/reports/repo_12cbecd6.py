import numpy as np

# Reproducing the bug with a single-element array
a = np.array([1.0+0.j])
print(f"Input array: {a}")
print(f"Input shape: {a.shape}")

try:
    result = np.fft.hfft(a)
    print(f"Result: {result}")
except ValueError as e:
    print(f"Error: {e}")

# Show that it works when n is explicitly provided
print("\nWith explicit n=1:")
result_with_n = np.fft.hfft(a, n=1)
print(f"Result with n=1: {result_with_n}")

# Compare with other FFT functions
print("\nComparing with other FFT functions on single-element array:")
print(f"fft(a): {np.fft.fft(a)}")
print(f"ifft(a): {np.fft.ifft(a)}")
print(f"rfft(a.real): {np.fft.rfft(a.real)}")
print(f"ihfft(a.real): {np.fft.ihfft(a.real)}")

# Show inverse relationship failure
print("\nInverse relationship test:")
b = np.array([1.0, 2.0])
print(f"Multi-element array: {b}")
print(f"ihfft(hfft(b)): {np.fft.ihfft(np.fft.hfft(b))}")

# Single element breaks the relationship
print(f"\nSingle-element array: {a.real}")
try:
    forward = np.fft.hfft(a.real)
    inverse = np.fft.ihfft(forward)
    print(f"ihfft(hfft(a)): {inverse}")
except ValueError as e:
    print(f"hfft fails with: {e}")
    print(f"But ihfft works: {np.fft.ihfft(a.real)}")