import numpy as np
import scipy.fft

# Test with a single-element array
x = np.array([5.0])
print(f"Original array: {x}")
print(f"Array shape: {x.shape}")

# Apply rfft - this should work
fft_result = scipy.fft.rfft(x)
print(f"rfft result: {fft_result}")
print(f"rfft result shape: {fft_result.shape}")

# Try to apply irfft - this will crash
try:
    roundtrip = scipy.fft.irfft(fft_result)
    print(f"irfft result: {roundtrip}")
except ValueError as e:
    print(f"Error occurred: {e}")

# Show that the workaround works
print("\nWorkaround with n=1:")
roundtrip_with_n = scipy.fft.irfft(fft_result, n=1)
print(f"irfft(fft_result, n=1) result: {roundtrip_with_n}")