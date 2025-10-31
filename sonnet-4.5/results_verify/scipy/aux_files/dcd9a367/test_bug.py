import numpy as np
import scipy.fft

# Test case from bug report
print("Testing with single-element complex array:")
x = np.array([1.0 + 0.0j])
print(f"Input: {x}")
print(f"Input shape: {x.shape}")

try:
    result = scipy.fft.irfft(x)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting with explicit n=1:")
try:
    result = scipy.fft.irfft(x, n=1)
    print(f"Result with n=1: {result}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting round-trip with rfft/irfft:")
real_input = np.array([1.0])
print(f"Original real input: {real_input}")
fft_result = scipy.fft.rfft(real_input)
print(f"After rfft: {fft_result}")
try:
    recovered = scipy.fft.irfft(fft_result)
    print(f"After irfft (without n): {recovered}")
except Exception as e:
    print(f"Error on irfft without n: {e}")

try:
    recovered = scipy.fft.irfft(fft_result, n=1)
    print(f"After irfft with n=1: {recovered}")
except Exception as e:
    print(f"Error on irfft with n=1: {e}")