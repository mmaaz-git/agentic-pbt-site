import numpy as np
import scipy.fft
from hypothesis import given, strategies as st, example
from numpy.testing import assert_allclose

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                min_size=1, max_size=1000),
       st.sampled_from([1, 4]))
@example(data=[0.0], dct_type=1)  # Force the failing case
def test_dct_self_inverse(data, dct_type):
    x = np.array(data)
    print(f"Testing with data={data[:5] if len(data) > 5 else data}, dct_type={dct_type}")
    try:
        transformed = scipy.fft.dct(x, type=dct_type, norm='ortho')
        result = scipy.fft.dct(transformed, type=dct_type, norm='ortho')
        assert_allclose(result, x, rtol=1e-10, atol=1e-10)
        print(f"  Success!")
    except Exception as e:
        print(f"  Failed: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    test_dct_self_inverse()