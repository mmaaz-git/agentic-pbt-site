from scipy.signal import windows
import numpy as np

# Manual reproduction - trying with underscores
print("--- Manual test with underscores (kaiser_bessel_derived) ---")
try:
    result = windows.get_window(('kaiser_bessel_derived', 8.6), 10, fftbins=False)
    print(f"SUCCESS: Got window with shape {result.shape}")
except ValueError as e:
    print(f"FAILED: {e}")

# Manual reproduction - trying with spaces
print("\n--- Manual test with spaces (kaiser bessel derived) ---")
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
print("\n--- Testing other window functions with underscores ---")

# general_gaussian
try:
    result = windows.get_window(('general_gaussian', 1.5, 2), 10, fftbins=False)
    print(f"general_gaussian with underscores: SUCCESS")
except ValueError as e:
    print(f"general_gaussian with underscores: FAILED - {e}")

# Try general gaussian with spaces
try:
    result = windows.get_window(('general gaussian', 1.5, 2), 10, fftbins=False)
    print(f"'general gaussian' with spaces: SUCCESS")
except ValueError as e:
    print(f"'general gaussian' with spaces: FAILED - {e}")

# general_cosine
try:
    result = windows.get_window(('general_cosine', [0.5, 0.5]), 10, fftbins=False)
    print(f"general_cosine with underscores: SUCCESS")
except ValueError as e:
    print(f"general_cosine with underscores: FAILED - {e}")

# Try general cosine with spaces
try:
    result = windows.get_window(('general cosine', [0.5, 0.5]), 10, fftbins=False)
    print(f"'general cosine' with spaces: SUCCESS")
except ValueError as e:
    print(f"'general cosine' with spaces: FAILED - {e}")

# general_hamming
try:
    result = windows.get_window(('general_hamming', 0.5), 10, fftbins=False)
    print(f"general_hamming with underscores: SUCCESS")
except ValueError as e:
    print(f"general_hamming with underscores: FAILED - {e}")

# Try general hamming with spaces
try:
    result = windows.get_window(('general hamming', 0.5), 10, fftbins=False)
    print(f"'general hamming' with spaces: SUCCESS")
except ValueError as e:
    print(f"'general hamming' with spaces: FAILED - {e}")

print("\n--- Direct function comparison ---")
# Verify they produce the same output when called correctly
# NOTE: kaiser_bessel_derived requires sym=True when called directly
w_direct = windows.kaiser_bessel_derived(10, beta=8.6, sym=True)
# get_window with fftbins=False creates a symmetric window
w_spaces = windows.get_window(('kaiser bessel derived', 8.6), 10, fftbins=False)
w_kbd = windows.get_window(('kbd', 8.6), 10, fftbins=False)

print(f"Direct vs spaces match: {np.allclose(w_direct, w_spaces)}")
print(f"Direct vs kbd match: {np.allclose(w_direct, w_kbd)}")
print(f"Spaces vs kbd match: {np.allclose(w_spaces, w_kbd)}")

# Let's check kaiser for completeness
print("\n--- Testing kaiser window (no underscores) ---")
try:
    result = windows.get_window(('kaiser', 8.6), 10, fftbins=False)
    print(f"kaiser: SUCCESS")
except ValueError as e:
    print(f"kaiser: FAILED - {e}")

# Let's examine what's in the mapping
print("\n--- Checking source code mapping ---")
import inspect
source_file = inspect.getfile(windows.get_window)
print(f"Source file: {source_file}")

# Let's verify the property-based test claim
print("\n--- Running hypothesis test manually ---")
beta_values = [0.1, 1.0, 5.0, 8.6, 10.0, 20.0]
for beta in beta_values:
    w_direct = windows.kaiser_bessel_derived(10, beta=beta, sym=True)
    try:
        w_from_get_window = windows.get_window(('kaiser_bessel_derived', beta), 10, fftbins=False)
        match = np.allclose(w_direct, w_from_get_window)
        print(f"Beta={beta}: {'PASS' if match else 'FAIL - values differ'}")
    except ValueError as e:
        print(f"Beta={beta}: FAIL - {e}")