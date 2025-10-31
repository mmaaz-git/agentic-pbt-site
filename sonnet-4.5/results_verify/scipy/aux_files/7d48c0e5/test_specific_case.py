import numpy as np
import scipy.fft

# Test the specific failing case
x = np.array([1.0])

print(f"Testing single-element array: {x}")
print(f"Array shape: {x.shape}")

try:
    print("\nTesting DCT type 1:")
    result = scipy.fft.dct(x, type=1)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

try:
    print("\nTesting DCT type 2:")
    result = scipy.fft.dct(x, type=2)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

try:
    print("\nTesting DCT type 3:")
    result = scipy.fft.dct(x, type=3)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

try:
    print("\nTesting DCT type 4:")
    result = scipy.fft.dct(x, type=4)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Also test with a 2-element array for type 1
print("\n" + "="*50)
x2 = np.array([1.0, 2.0])
print(f"\nTesting 2-element array: {x2}")
print(f"Array shape: {x2.shape}")

try:
    print("\nTesting DCT type 1 with 2-element array:")
    result = scipy.fft.dct(x2, type=1)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")