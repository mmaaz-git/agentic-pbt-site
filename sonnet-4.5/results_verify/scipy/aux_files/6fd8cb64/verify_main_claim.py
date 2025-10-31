import numpy as np
import scipy.fftpack as fftpack

# Let's focus on the main claim in the bug report: Even-length arrays fail round-trip
print("Testing the main claim: Even-length arrays fail round-trip even when sum=0")
print("=" * 60)

# Test 1: Simple even-length arrays
for n in [2, 4, 6, 8, 10]:
    x = np.random.randn(n)
    x = x - np.mean(x)  # Ensure sum is zero

    y = fftpack.hilbert(x)
    result = fftpack.ihilbert(y)

    works = np.allclose(result, x, atol=1e-10, rtol=1e-10)
    print(f"n={n:2d}: sum={np.sum(x):+.2e}, round-trip works: {works}")

print("\nTesting odd-length arrays for comparison:")
print("=" * 60)

# Test 2: Simple odd-length arrays
for n in [3, 5, 7, 9, 11]:
    x = np.random.randn(n)
    x = x - np.mean(x)  # Ensure sum is zero

    y = fftpack.hilbert(x)
    result = fftpack.ihilbert(y)

    works = np.allclose(result, x, atol=1e-10, rtol=1e-10)
    print(f"n={n:2d}: sum={np.sum(x):+.2e}, round-trip works: {works}")

print("\nDetailed example with even n=2:")
print("=" * 60)
x = np.array([1.0, -1.0])
print(f"Input x: {x}")
print(f"Sum: {np.sum(x)}")
y = fftpack.hilbert(x)
print(f"After hilbert: {y}")
result = fftpack.ihilbert(y)
print(f"After ihilbert: {result}")
print(f"Difference: {result - x}")

print("\nDetailed example with odd n=3:")
print("=" * 60)
x = np.array([1.0, -2.0, 1.0])
print(f"Input x: {x}")
print(f"Sum: {np.sum(x)}")
y = fftpack.hilbert(x)
print(f"After hilbert: {y}")
result = fftpack.ihilbert(y)
print(f"After ihilbert: {result}")
print(f"Difference: {result - x}")