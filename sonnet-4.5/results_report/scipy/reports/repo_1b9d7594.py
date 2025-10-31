import numpy as np
import scipy.fft

# Test case with single-element array
x = np.array([5.0])
print(f"Input array: {x}")
print(f"Input shape: {x.shape}")

# Perform rfft
rfft_out = scipy.fft.rfft(x)
print(f"\nrfft output: {rfft_out}")
print(f"rfft output shape: {rfft_out.shape}")

# Try to perform irfft without specifying n (this should fail)
print("\nAttempting irfft without n parameter...")
try:
    irfft_out = scipy.fft.irfft(rfft_out)
    print(f"irfft output: {irfft_out}")
except ValueError as e:
    print(f"ERROR: {e}")

# Now try with n specified
print("\nAttempting irfft with n=1...")
irfft_out_with_n = scipy.fft.irfft(rfft_out, n=1)
print(f"irfft output with n=1: {irfft_out_with_n}")
print(f"Matches original? {np.allclose(irfft_out_with_n, x)}")