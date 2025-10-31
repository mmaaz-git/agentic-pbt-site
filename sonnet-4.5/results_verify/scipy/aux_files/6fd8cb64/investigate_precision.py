import numpy as np
import scipy.fftpack as fftpack

# Test the failing odd-length case from the property test
x_odd = np.array([7.23362913e+09] * 19)
x_odd = x_odd - np.mean(x_odd)  # Make sum zero

print("Odd-length array (n=19) with large values:")
print(f"  Sum: {np.sum(x_odd)}")
print(f"  Values are all: {x_odd[0]}")

y_odd = fftpack.hilbert(x_odd)
result_odd = fftpack.ihilbert(y_odd)

print(f"  Round-trip works: {np.allclose(result_odd, x_odd, atol=1e-10, rtol=1e-10)}")
print(f"  Round-trip works (looser tolerance): {np.allclose(result_odd, x_odd, atol=1e-6, rtol=1e-6)}")
print(f"  Max error: {np.max(np.abs(result_odd - x_odd))}")

# Test with smaller values
x_small = np.array([1.0] * 19)
x_small = x_small - np.mean(x_small)

print("\nOdd-length array (n=19) with small values:")
print(f"  Sum: {np.sum(x_small)}")
print(f"  Values are all: {x_small[0]}")

y_small = fftpack.hilbert(x_small)
result_small = fftpack.ihilbert(y_small)

print(f"  Round-trip works: {np.allclose(result_small, x_small, atol=1e-10, rtol=1e-10)}")
print(f"  Max error: {np.max(np.abs(result_small - x_small))}")