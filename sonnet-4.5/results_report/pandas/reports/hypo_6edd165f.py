from hypothesis import given, strategies as st
import numpy as np
import pandas as pd

@given(st.lists(st.complex_numbers(allow_nan=False, allow_infinity=False), min_size=1))
def test_complex_hashing_consistency(values):
    arr64 = np.array(values, dtype=np.complex64)
    arr128 = np.array(values, dtype=np.complex128)

    hash64 = pd.util.hash_array(arr64)
    hash128 = pd.util.hash_array(arr128)

    hash64_real = pd.util.hash_array(arr64.real)
    hash64_imag = pd.util.hash_array(arr64.imag)
    expected64 = hash64_real + 23 * hash64_imag

    assert np.array_equal(hash64, expected64), \
        "complex64 should use same formula as complex128: hash_real + 23 * hash_imag"

if __name__ == "__main__":
    test_complex_hashing_consistency()