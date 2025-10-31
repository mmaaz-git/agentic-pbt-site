from hypothesis import given, strategies as st
from scipy.signal import windows
import numpy as np

# First let's test the hypothesis test from the bug report
@given(st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False))
def test_get_window_recognizes_function_names(beta):
    w_direct = windows.kaiser_bessel_derived(10, beta=beta, sym=True)

    try:
        w_from_get_window = windows.get_window(('kaiser_bessel_derived', beta), 10, fftbins=False)
        assert np.allclose(w_direct, w_from_get_window), \
            "get_window should accept the actual function name 'kaiser_bessel_derived'"
        return "PASS"
    except ValueError as e:
        return f"FAIL: {e}"

# Test with a single value
print("Testing hypothesis test with beta=8.6:")
print(test_get_window_recognizes_function_names(8.6))

# Manual reproduction - trying with underscores
print("\n--- Manual test with underscores ---")
try:
    result = windows.get_window(('kaiser_bessel_derived', 8.6), 10, fftbins=False)
    print(f"SUCCESS: Got window with shape {result.shape}")
except ValueError as e:
    print(f"FAILED: {e}")

# Manual reproduction - trying with spaces
print("\n--- Manual test with spaces ---")
try:
    result = windows.get_window(('kaiser bessel derived', 8.6), 10, fftbins=False)
    print(f"SUCCESS: Got window with shape {result.shape}")
except ValueError as e:
    print(f"FAILED: {e}")

# Manual reproduction - trying with alias kbd
print("\n--- Manual test with alias 'kbd' ---")
try:
    result = windows.get_window(('kbd', 8.6), 10, fftbins=False)
    print(f"SUCCESS: Got window with shape {result.shape}")
except ValueError as e:
    print(f"FAILED: {e}")

# Let's also check other windows with underscores
print("\n--- Testing other window functions ---")
# general_gaussian
try:
    result = windows.get_window(('general_gaussian', 1.5, 2), 10, fftbins=False)
    print(f"general_gaussian with underscores: SUCCESS")
except ValueError as e:
    print(f"general_gaussian with underscores: FAILED - {e}")

# general_cosine
try:
    result = windows.get_window(('general_cosine', [0.5, 0.5]), 10, fftbins=False)
    print(f"general_cosine with underscores: SUCCESS")
except ValueError as e:
    print(f"general_cosine with underscores: FAILED - {e}")

# general_hamming
try:
    result = windows.get_window(('general_hamming', 0.5), 10, fftbins=False)
    print(f"general_hamming with underscores: SUCCESS")
except ValueError as e:
    print(f"general_hamming with underscores: FAILED - {e}")

print("\n--- Direct function comparison ---")
# Verify they produce the same output when called correctly
w_direct = windows.kaiser_bessel_derived(10, beta=8.6, sym=False)
w_spaces = windows.get_window(('kaiser bessel derived', 8.6), 10, fftbins=False)
w_kbd = windows.get_window(('kbd', 8.6), 10, fftbins=False)

print(f"Direct vs spaces match: {np.allclose(w_direct, w_spaces)}")
print(f"Direct vs kbd match: {np.allclose(w_direct, w_kbd)}")
print(f"Spaces vs kbd match: {np.allclose(w_spaces, w_kbd)}")