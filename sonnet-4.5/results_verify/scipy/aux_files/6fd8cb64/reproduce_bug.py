import numpy as np
import scipy.fftpack as fftpack

# Test even-length array
x_even = np.array([-0.5, 0.5])
print(f"Even-length array (n=2):")
print(f"  Input:  {x_even}")
print(f"  Sum of input: {np.sum(x_even)}")

y_even = fftpack.hilbert(x_even)
print(f"  After hilbert:  {y_even}")

result_even = fftpack.ihilbert(y_even)
print(f"  After ihilbert: {result_even}")
print(f"  Round-trip successful: {np.allclose(result_even, x_even)}")

# Test odd-length array
x_odd = np.array([-1.0, 0.0, 1.0])
print(f"\nOdd-length array (n=3):")
print(f"  Input:  {x_odd}")
print(f"  Sum of input: {np.sum(x_odd)}")

y_odd = fftpack.hilbert(x_odd)
print(f"  After hilbert:  {y_odd}")

result_odd = fftpack.ihilbert(y_odd)
print(f"  After ihilbert: {result_odd}")
print(f"  Round-trip successful: {np.allclose(result_odd, x_odd)}")

# Test more even-length examples
print("\nTesting more even-length arrays:")
for n in [4, 6, 8, 10]:
    x = np.random.randn(n)
    x = x - np.mean(x)  # Ensure sum is zero
    y = fftpack.hilbert(x)
    result = fftpack.ihilbert(y)
    success = np.allclose(result, x)
    print(f"  n={n}: sum={np.sum(x):.2e}, round-trip successful: {success}")

# Test more odd-length examples
print("\nTesting more odd-length arrays:")
for n in [3, 5, 7, 9]:
    x = np.random.randn(n)
    x = x - np.mean(x)  # Ensure sum is zero
    y = fftpack.hilbert(x)
    result = fftpack.ihilbert(y)
    success = np.allclose(result, x)
    print(f"  n={n}: sum={np.sum(x):.2e}, round-trip successful: {success}")