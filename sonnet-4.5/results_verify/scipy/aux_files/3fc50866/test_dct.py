import numpy as np
import scipy.fft

# First, reproduce the exact bug from the report
print("=== Testing the reported bug ===")
print("Testing scipy.fft.dct with type=1 on single-element array:")
try:
    x = np.array([1.0])
    result = scipy.fft.dct(x, type=1)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n=== Testing idct with type=1 on single-element array ===")
try:
    x = np.array([1.0])
    result = scipy.fft.idct(x, type=1)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n=== Testing other DCT types with single-element arrays ===")
x = np.array([1.0])
for dct_type in [2, 3, 4]:
    print(f"\nType {dct_type}:")
    try:
        result_dct = scipy.fft.dct(x, type=dct_type)
        print(f"  dct: Success! Result: {result_dct}")
        result_idct = scipy.fft.idct(x, type=dct_type)
        print(f"  idct: Success! Result: {result_idct}")
        # Test roundtrip
        roundtrip = scipy.fft.idct(scipy.fft.dct(x, type=dct_type), type=dct_type)
        print(f"  Roundtrip: {roundtrip}, matches input: {np.allclose(roundtrip, x)}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

print("\n=== Testing DCT-I with 2-element array ===")
x2 = np.array([1.0, 2.0])
try:
    result = scipy.fft.dct(x2, type=1)
    print(f"Success! Result: {result}")
    roundtrip = scipy.fft.idct(scipy.fft.dct(x2, type=1), type=1)
    print(f"Roundtrip: {roundtrip}, matches input: {np.allclose(roundtrip, x2)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Run the test with a specific failing example
print("\n=== Testing with the reported failing input: x=[0.0], dct_type=1 ===")
try:
    x_arr = np.array([0.0])
    result = scipy.fft.idct(scipy.fft.dct(x_arr, type=1), type=1)
    if np.allclose(result, x_arr, rtol=1e-9, atol=1e-8):
        print("Test passed!")
    else:
        print(f"Test failed: roundtrip did not match. Input: {x_arr}, Result: {result}")
except Exception as e:
    print(f"Test failed with: {type(e).__name__}: {e}")