import numpy as np
import scipy.fft

# Simple reproduction from the bug report
x = np.array([42.0])

print("Testing DCT Type I with size-1 array:")
try:
    result = scipy.fft.dct(x, type=1)
    print(f"Result: {result}")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
except Exception as e:
    print(f"Other error ({type(e).__name__}): {e}")
