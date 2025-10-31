import numpy as np
import scipy.fftpack as fftpack

print("Testing Hilbert round-trip for even vs odd length arrays:")

for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    x = np.arange(n, dtype=float)
    x = x - np.mean(x)

    result = fftpack.hilbert(fftpack.ihilbert(x))
    matches = np.allclose(result, x, atol=1e-10)

    print(f"Length {n} ({'even' if n % 2 == 0 else 'odd '}): {'PASS' if matches else 'FAIL'}")

print("\nSpecific example with length 2:")
x = np.array([-1.0, 1.0])
print(f"x = {x}, sum(x) = {np.sum(x)}")
result = fftpack.hilbert(fftpack.ihilbert(x))
print(f"hilbert(ihilbert(x)) = {result}")
print(f"Expected: {x}")
print(f"Match: {np.allclose(result, x)}")

print("\nAdditional tests:")
print("Testing with various even-length arrays where sum(x) == 0:")
test_cases = [
    np.array([-1.0, 1.0]),
    np.array([-2.0, -1.0, 1.0, 2.0]),
    np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
]

for x in test_cases:
    print(f"\nArray: {x}")
    print(f"Sum: {np.sum(x)}")
    result = fftpack.hilbert(fftpack.ihilbert(x))
    print(f"Result: {result}")
    print(f"Match: {np.allclose(result, x, atol=1e-10)}")