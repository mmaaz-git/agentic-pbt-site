#!/usr/bin/env python3
"""Test how other scipy.signal functions handle empty arrays"""

import numpy as np
import scipy.signal as signal
import traceback

# Test various signal processing functions with empty arrays
x = np.array([], dtype=np.float64)

print("Testing scipy.signal functions with empty array:")
print("=" * 50)

# Test resample_poly
print("\n1. resample_poly (resampling by ratio):")
try:
    y = signal.resample_poly(x, 2, 3)
    print(f"   Result: {y}, shape: {y.shape}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Test decimate
print("\n2. decimate (downsample):")
try:
    y = signal.decimate(x, 2)
    print(f"   Result: {y}, shape: {y.shape}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Test convolve
print("\n3. convolve:")
try:
    y = signal.convolve(x, np.array([1, 2, 3]))
    print(f"   Result: {y}, shape: {y.shape}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Test correlate
print("\n4. correlate:")
try:
    y = signal.correlate(x, np.array([1, 2, 3]))
    print(f"   Result: {y}, shape: {y.shape}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Test fftconvolve
print("\n5. fftconvolve:")
try:
    y = signal.fftconvolve(x, np.array([1, 2, 3]))
    print(f"   Result: {y}, shape: {y.shape}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Test hilbert
print("\n6. hilbert (analytic signal):")
try:
    y = signal.hilbert(x)
    print(f"   Result: {y}, shape: {y.shape}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Test detrend
print("\n7. detrend:")
try:
    y = signal.detrend(x)
    print(f"   Result: {y}, shape: {y.shape}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Test savgol_filter
print("\n8. savgol_filter (requires at least window_length samples):")
try:
    y = signal.savgol_filter(x, 5, 2)
    print(f"   Result: {y}, shape: {y.shape}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Test medfilt
print("\n9. medfilt (median filter):")
try:
    y = signal.medfilt(x)
    print(f"   Result: {y}, shape: {y.shape}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Test filtfilt
print("\n10. filtfilt (zero-phase filter):")
try:
    b, a = signal.butter(2, 0.5)
    y = signal.filtfilt(b, a, x)
    print(f"   Result: {y}, shape: {y.shape}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")