import numpy as np

a = np.array([0.])
print(f"Input array: {a}")

rfft_result = np.fft.rfft(a)
print(f"rfft result: {rfft_result}")

# Test with explicit n=1 (what the proposed fix would do)
result = np.fft.irfft(rfft_result, n=1)
print(f"irfft with n=1: {result}")

# Test the round-trip
print(f"\nRound-trip test:")
print(f"Original: {a}")
print(f"After rfft->irfft(n=1): {result}")
print(f"Are they equal? {np.allclose(a, result)}")