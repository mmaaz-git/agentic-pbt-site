import numpy as np

a = np.array([0.])
print(f"Input array: {a}")
print(f"Input shape: {a.shape}")

rfft_result = np.fft.rfft(a)
print(f"rfft result: {rfft_result}")
print(f"rfft result shape: {rfft_result.shape}")

try:
    result = np.fft.irfft(rfft_result)
    print(f"irfft result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")