import numpy as np
import scipy.fft

x = np.array([1.0])

print("Testing consistency across DCT/DST types with single-element array:")
print(f"Input array: {x}")
print()

# Test DCT types
for dct_type in [1, 2, 3, 4]:
    try:
        result = scipy.fft.dct(x, type=dct_type, norm='ortho')
        print(f"DCT type {dct_type}: Success! Result = {result}")
    except Exception as e:
        print(f"DCT type {dct_type}: Error - {type(e).__name__}: {e}")

print()

# Test DST types
for dst_type in [1, 2, 3, 4]:
    try:
        result = scipy.fft.dst(x, type=dst_type, norm='ortho')
        print(f"DST type {dst_type}: Success! Result = {result}")
    except Exception as e:
        print(f"DST type {dst_type}: Error - {type(e).__name__}: {e}")

print("\nTesting with 2-element array for DCT type 1:")
x2 = np.array([1.0, 2.0])
try:
    result = scipy.fft.dct(x2, type=1, norm='ortho')
    print(f"DCT type 1 with 2 elements: Success! Result = {result}")
except Exception as e:
    print(f"DCT type 1 with 2 elements: Error - {type(e).__name__}: {e}")