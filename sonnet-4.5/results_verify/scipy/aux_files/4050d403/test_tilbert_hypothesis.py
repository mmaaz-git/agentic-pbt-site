from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy import fftpack


@settings(max_examples=500)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6),
                min_size=2, max_size=100))
def test_tilbert_itilbert_roundtrip(x_list):
    x = np.array(x_list)
    x = x - x.mean()
    assume(np.sum(np.abs(x)) > 1e-6)

    h_param = 0.5

    t = fftpack.tilbert(x, h=h_param)
    it = fftpack.itilbert(t, h=h_param)

    assert np.allclose(it, x, rtol=1e-3, atol=1e-5)

if __name__ == "__main__":
    test_tilbert_itilbert_roundtrip()
    print("All tests passed!")