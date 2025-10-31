import numpy as np
import scipy.fft

x = np.array([1.0])

print("Testing other DCT types with size-1 arrays:")
for dct_type in [2, 3, 4]:
    try:
        result = scipy.fft.dct(x, type=dct_type)
        print(f"DCT type {dct_type}: {result}")
    except Exception as e:
        print(f"DCT type {dct_type} error: {e}")
