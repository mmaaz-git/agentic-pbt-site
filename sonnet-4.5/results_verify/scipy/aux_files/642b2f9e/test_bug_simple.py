import numpy as np
import scipy.fftpack as fftpack

# Reproducing the bug with the simple test case
x = np.array([1., 2., 3., 4.])
result = fftpack.ihilbert(fftpack.hilbert(x))

print(f"Input:    {x}")
print(f"Output:   {result}")
print(f"Expected: {x}")
print(f"Match:    {np.allclose(result, x)}")
print(f"Difference: {result - x}")