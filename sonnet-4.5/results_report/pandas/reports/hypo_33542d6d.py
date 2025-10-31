import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.util.hashing import hash_array

@given(st.lists(st.complex_numbers(allow_nan=False, allow_infinity=False, min_magnitude=1e-10, max_magnitude=1e10), min_size=1))
@settings(max_examples=500)
def test_hash_array_complex64_vs_complex128(values):
    arr64 = np.array(values, dtype=np.complex64)
    arr128 = arr64.astype(np.complex128)

    hash64 = hash_array(arr64)

    real64 = hash_array(arr64.real)
    imag64 = hash_array(arr64.imag)
    expected64 = real64 + 23 * imag64

    assert np.array_equal(hash64, expected64)

# Run the test
test_hash_array_complex64_vs_complex128()