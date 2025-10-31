from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy import fftpack


@settings(max_examples=500)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6),
                min_size=1, max_size=100),
       st.integers(min_value=1, max_value=4))
def test_dct_orthogonality(x_list, dct_type):
    x = np.array(x_list)
    dct_x = fftpack.dct(x, type=dct_type, norm='ortho')
    energy_orig = np.sum(x**2)
    energy_dct = np.sum(dct_x**2)
    assert np.isclose(energy_orig, energy_dct, rtol=1e-4, atol=1e-6)

# Run the test
print("Running hypothesis test...")
try:
    test_dct_orthogonality()
    print("All tests passed!")
except Exception as e:
    print(f"Test failed with error: {e}")

# Now specifically test the failing case mentioned
print("\nTesting specific failing case: x_list=[0.0], dct_type=1")
try:
    x = np.array([0.0])
    dct_x = fftpack.dct(x, type=1, norm='ortho')
    print(f"Result: {dct_x}")
except Exception as e:
    print(f"Error ({type(e).__name__}): {e}")