import numpy as np
import scipy.fftpack as fftpack

# Test the documented condition: "If sum(x, axis=0) == 0 then hilbert(ihilbert(x)) == x"
print("Testing documented condition:")

# Create arrays with sum = 0
test_cases = [
    np.array([1., -1.]),
    np.array([1., -2., 1.]),
    np.array([1., -1., 1., -1.]),
    np.array([2., -1., -1.]),
    np.array([1., 2., -3.])
]

for x in test_cases:
    sum_x = np.sum(x)
    result = fftpack.hilbert(fftpack.ihilbert(x))
    match = np.allclose(result, x, rtol=1e-10, atol=1e-12)
    print(f"\nInput: {x}")
    print(f"Sum: {sum_x}")
    print(f"hilbert(ihilbert(x)): {result}")
    print(f"Match: {match}")

# Also test the reverse: ihilbert(hilbert(x))
print("\n\nTesting reverse order (ihilbert(hilbert(x))) with sum=0:")
for x in test_cases:
    sum_x = np.sum(x)
    result = fftpack.ihilbert(fftpack.hilbert(x))
    match = np.allclose(result, x, rtol=1e-10, atol=1e-12)
    print(f"\nInput: {x}")
    print(f"Sum: {sum_x}")
    print(f"ihilbert(hilbert(x)): {result}")
    print(f"Match: {match}")