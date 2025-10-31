import numpy as np
import scipy.fft
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as st_np

@given(st_np.arrays(
    dtype=np.float64,
    shape=st_np.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=200),
    elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
))
@settings(max_examples=500)
def test_dct_idct_roundtrip(x):
    for dct_type in [1, 2, 3, 4]:
        print(f"Testing DCT type {dct_type} with array shape {x.shape}")
        try:
            result = scipy.fft.idct(scipy.fft.dct(x, type=dct_type), type=dct_type)
            max_val = max(np.max(np.abs(x)), 1.0)
            atol = max_val * 1e-12
            assert np.allclose(result, x, rtol=1e-12, atol=atol), \
                f"DCT type {dct_type} roundtrip failed"
        except Exception as e:
            print(f"Failed on DCT type {dct_type} with array: {x}")
            print(f"Error: {e}")
            raise

# Run the test
test_dct_idct_roundtrip()