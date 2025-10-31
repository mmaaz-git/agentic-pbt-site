import numpy as np

# Test the simple case
a = np.array([1.0])
print(f"Testing hfft with single-element array: {a}")
try:
    result = np.fft.hfft(a)
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with explicit n parameter as mentioned in the workaround
print("\nTesting hfft with single-element array and explicit n=2:")
try:
    result = np.fft.hfft(a, n=2)
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")