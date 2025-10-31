import numpy as np
import scipy.fftpack as fftpack
from hypothesis import given, strategies as st, settings, assume

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=20),
       st.integers(min_value=1, max_value=3))
@settings(max_examples=500)
def test_diff_roundtrip_zero_mean(lst, order):
    x = np.array(lst)
    x = x - np.mean(x)
    assume(np.abs(np.sum(x)) < 1e-10)

    diff_x = fftpack.diff(x, order=order)
    roundtrip = fftpack.diff(diff_x, order=-order)
    assert np.allclose(roundtrip, x, atol=1e-4), f"diff round-trip failed for order={order}: {roundtrip} vs {x}"

if __name__ == "__main__":
    test_diff_roundtrip_zero_mean()