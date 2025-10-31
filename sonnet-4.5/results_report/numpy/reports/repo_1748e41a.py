import numpy as np

a = np.array([0.])
print(f"Original array: {a}")
print(f"Original array shape: {a.shape}")

rfft_result = np.fft.rfft(a)
print(f"rfft result: {rfft_result}")
print(f"rfft result shape: {rfft_result.shape}")

print("\nAttempting irfft without n parameter...")
try:
    result = np.fft.irfft(rfft_result)
    print(f"irfft result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nAttempting irfft with n=1...")
try:
    result = np.fft.irfft(rfft_result, n=1)
    print(f"irfft result with n=1: {result}")
    print(f"Does it match original? {np.allclose(result, a)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")