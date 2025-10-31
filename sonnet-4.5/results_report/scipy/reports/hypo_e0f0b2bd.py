import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.signal import windows


window_functions_no_params = [
    'boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
    'blackmanharris', 'flattop', 'bartlett', 'barthann',
    'hamming', 'cosine', 'hann', 'lanczos', 'tukey'
]


@given(
    window_name=st.sampled_from(window_functions_no_params),
    M=st.integers(min_value=1, max_value=10000)
)
@settings(max_examples=500)
def test_normalization_property(window_name, M):
    window = windows.get_window(window_name, M, fftbins=True)
    max_val = np.max(np.abs(window))
    assert max_val <= 1.0 + 1e-10, f"{window_name} with M={M} has max value {max_val} > 1.0"

if __name__ == "__main__":
    test_normalization_property()