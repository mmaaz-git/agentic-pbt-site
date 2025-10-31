from hypothesis import given, strategies as st, assume, settings
import numpy as np
import scipy.signal as signal

@settings(max_examples=500)
@given(
    b=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e4, max_value=1e4), min_size=1, max_size=5),
    a=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e4, max_value=1e4), min_size=1, max_size=5)
)
def test_tf2ss_ss2tf_roundtrip(b, a):
    b_arr = np.array(b)
    a_arr = np.array(a)
    assume(np.abs(a_arr[0]) > 1e-10)
    assume(np.abs(b_arr[0]) > 1e-10)
    assume(len(a_arr) >= len(b_arr))

    A, B, C, D = signal.tf2ss(b_arr, a_arr)
    b_reconstructed, a_reconstructed = signal.ss2tf(A, B, C, D)

    # Normalize and compare
    b_norm = b_arr / b_arr[0]
    a_norm = a_arr / a_arr[0]

    if b_reconstructed.ndim == 2:
        b_reconstructed = b_reconstructed[0]

    b_recon_norm = b_reconstructed / b_reconstructed[0]
    a_recon_norm = a_reconstructed / a_reconstructed[0]

    assert len(b_recon_norm) == len(b_norm), f"Length mismatch: reconstructed {len(b_recon_norm)} vs original {len(b_norm)}"
    np.testing.assert_allclose(b_recon_norm, b_norm, rtol=1e-4, atol=1e-6)

if __name__ == "__main__":
    test_tf2ss_ss2tf_roundtrip()