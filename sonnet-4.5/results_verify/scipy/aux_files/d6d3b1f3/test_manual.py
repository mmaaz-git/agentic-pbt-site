import numpy as np
import scipy.signal.windows as windows

M = 10

direct = windows.hann(M)
via_get_window = windows.get_window('hann', M)

print(f"hann({M}):")
print(direct)

print(f"\nget_window('hann', {M}):")
print(via_get_window)

print(f"\nAre they equal? {np.allclose(direct, via_get_window)}")

# Let's also check what defaults are being used
print("\n--- Checking defaults ---")
direct_with_sym_true = windows.hann(M, sym=True)
direct_with_sym_false = windows.hann(M, sym=False)
get_window_with_fftbins_true = windows.get_window('hann', M, fftbins=True)
get_window_with_fftbins_false = windows.get_window('hann', M, fftbins=False)

print(f"hann({M}) == hann({M}, sym=True): {np.allclose(direct, direct_with_sym_true)}")
print(f"hann({M}) == hann({M}, sym=False): {np.allclose(direct, direct_with_sym_false)}")
print(f"get_window('hann', {M}) == get_window('hann', {M}, fftbins=True): {np.allclose(via_get_window, get_window_with_fftbins_true)}")
print(f"get_window('hann', {M}) == get_window('hann', {M}, fftbins=False): {np.allclose(via_get_window, get_window_with_fftbins_false)}")

print(f"\nhann({M}, sym=False) == get_window('hann', {M}, fftbins=True): {np.allclose(direct_with_sym_false, get_window_with_fftbins_true)}")
print(f"hann({M}, sym=True) == get_window('hann', {M}, fftbins=False): {np.allclose(direct_with_sym_true, get_window_with_fftbins_false)}")