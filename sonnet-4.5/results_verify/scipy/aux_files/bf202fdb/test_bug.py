import numpy as np
import scipy.fft

print("Testing DCT type 1 with single element array:")
x = np.array([0.0])
try:
    result = scipy.fft.dct(x, type=1, norm='ortho')
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting DCT type 1 with single non-zero element:")
x = np.array([1.0])
try:
    result = scipy.fft.dct(x, type=1, norm='ortho')
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")