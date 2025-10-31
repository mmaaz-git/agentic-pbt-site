import numpy as np
from scipy import fftpack

x = np.array([0.0])

print(f"Input: {x}")
print(f"Length: {len(x)}")

print("\nTesting DCT Type-1 with single element:")
try:
    result = fftpack.dct(x, type=1, norm='ortho')
    print(f"Result: {result}")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
except Exception as e:
    print(f"Other Exception ({type(e).__name__}): {e}")

print("\nTesting DCT Type-2 with single element (for comparison):")
result2 = fftpack.dct(x, type=2, norm='ortho')
print(f"Result: {result2}")

print("\nTesting DCT Type-1 with two elements:")
x2 = np.array([0.0, 1.0])
try:
    result_two = fftpack.dct(x2, type=1, norm='ortho')
    print(f"Result: {result_two}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Additional Tests ---")

print("\nTesting DCT Type-1 with norm=None (as per docs requirement):")
try:
    result = fftpack.dct(x, type=1, norm=None)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error ({type(e).__name__}): {e}")

print("\nTesting all DCT types with single element:")
for dct_type in [1, 2, 3, 4]:
    try:
        result = fftpack.dct(x, type=dct_type, norm='ortho')
        print(f"DCT Type-{dct_type}: {result}")
    except Exception as e:
        print(f"DCT Type-{dct_type}: Error ({type(e).__name__}): {e}")