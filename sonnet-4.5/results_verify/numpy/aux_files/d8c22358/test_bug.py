import numpy as np

# Reproduce the bug as described
a = np.fft.ifft([1.0, 2.0, 3.0])
n = 2 * len(a) - 2

result = np.fft.ihfft(np.fft.hfft(a, n))

print(f"Original a: {a}")
print(f"n value: {n}")
print(f"Expected: {a}")
print(f"Got:      {result}")
print(f"Match: {np.allclose(result, a)}")
print(f"Difference: {result - a}")

# Also test the odd case for comparison
n_odd = 2 * len(a) - 1
result_odd = np.fft.ihfft(np.fft.hfft(a, n_odd))
print(f"\nOdd case (n={n_odd}):")
print(f"Expected: {a}")
print(f"Got:      {result_odd}")
print(f"Match: {np.allclose(result_odd, a)}")
print(f"Difference: {result_odd - a}")