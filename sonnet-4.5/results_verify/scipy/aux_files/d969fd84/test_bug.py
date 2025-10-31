import numpy as np
import scipy.fft
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst

# First test the simple reproduction case
print("Testing simple reproduction case:")
x = np.array([0.])
print(f"Input: {x}")

# Test each DCT type
for dct_type in [1, 2, 3, 4]:
    try:
        result = scipy.fft.dct(x, type=dct_type)
        print(f"DCT type {dct_type}: Success - Result = {result}")
    except Exception as e:
        print(f"DCT type {dct_type}: Error - {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Now run the hypothesis test
print("Running the property-based test:")

@given(npst.arrays(
    dtype=np.float64,
    shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
    elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e7, max_value=1e7)
))
@settings(max_examples=500)
def test_dct_idct_roundtrip(x):
    for dct_type in [1, 2, 3, 4]:
        try:
            transformed = scipy.fft.dct(x, type=dct_type)
            result = scipy.fft.idct(transformed, type=dct_type)
            assert np.allclose(result, x, rtol=1e-8, atol=1e-8), f"DCT type {dct_type} roundtrip failed"
        except RuntimeError as e:
            if dct_type == 1 and len(x) == 1:
                # This is expected based on documentation
                pass
            else:
                raise

try:
    test_dct_idct_roundtrip()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Test with various single-element arrays
print("\n" + "="*50 + "\n")
print("Testing various single-element arrays:")
test_values = [0., 1., -1., 5., 100.]
for val in test_values:
    x = np.array([val])
    print(f"\nInput: array([{val}])")
    for dct_type in [1, 2, 3, 4]:
        try:
            result = scipy.fft.dct(x, type=dct_type)
            print(f"  DCT type {dct_type}: {result}")
        except RuntimeError as e:
            print(f"  DCT type {dct_type}: RuntimeError - {e}")