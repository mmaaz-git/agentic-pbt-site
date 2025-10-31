import numpy as np
import scipy.fft

print("Testing DST type 1 with single-element array:")
x = np.array([1.0])
try:
    result = scipy.fft.dst(x, type=1, norm='ortho')
    print(f"Success! Result = {result}")
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting DST type 1 with zero single-element:")
x = np.array([0.0])
try:
    result = scipy.fft.dst(x, type=1, norm='ortho')
    print(f"Success! Result = {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")