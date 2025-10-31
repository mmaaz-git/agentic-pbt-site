import numpy as np

# Demonstrate the crash with numpy.fft.hfft on single-element array
a = np.array([1.0])
try:
    result = np.fft.hfft(a)
    print(f"Success: hfft([1.0]) = {result}")
except ValueError as e:
    print(f"Error with hfft([1.0]): {e}")

# Show that the workaround works
try:
    result_with_n = np.fft.hfft(a, n=2)
    print(f"Success with explicit n: hfft([1.0], n=2) = {result_with_n}")
except ValueError as e:
    print(f"Error with hfft([1.0], n=2): {e}")

# Test other FFT functions for comparison
print("\nComparison with other FFT functions:")
functions_to_test = [
    ('fft', np.fft.fft),
    ('ifft', np.fft.ifft),
    ('rfft', np.fft.rfft),
    ('irfft', np.fft.irfft),
    ('hfft', np.fft.hfft),
    ('ihfft', np.fft.ihfft)
]

for name, func in functions_to_test:
    try:
        result = func(a)
        print(f"{name}: SUCCESS - returns {result}")
    except ValueError as e:
        print(f"{name}: FAILS - {e}")