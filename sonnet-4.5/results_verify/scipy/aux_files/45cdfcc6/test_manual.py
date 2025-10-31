import numpy as np
import scipy.fft

x = np.array([5.0])
print(f"Input array: {x}")
print(f"Input shape: {x.shape}")

fft_result = scipy.fft.rfft(x)
print(f"rfft result: {fft_result}")
print(f"rfft result shape: {fft_result.shape}")

try:
    roundtrip = scipy.fft.irfft(fft_result)
    print(f"Roundtrip successful: {roundtrip}")
except ValueError as e:
    print(f"Error: {e}")

# Test the workaround
print("\nTesting workaround with n=1:")
try:
    roundtrip_workaround = scipy.fft.irfft(fft_result, n=1)
    print(f"Workaround successful: {roundtrip_workaround}")
    print(f"Original vs roundtrip: {x} vs {roundtrip_workaround}")
except Exception as e:
    print(f"Workaround failed: {e}")