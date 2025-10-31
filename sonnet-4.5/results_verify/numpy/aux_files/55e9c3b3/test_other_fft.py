import numpy as np

# Test various FFT functions with single-element arrays
a = np.array([1.0])
a_complex = np.array([1.0 + 0j])

print("Testing various FFT functions with single-element arrays:\n")

functions_to_test = [
    ("fft", lambda: np.fft.fft(a_complex)),
    ("ifft", lambda: np.fft.ifft(a_complex)),
    ("rfft", lambda: np.fft.rfft(a)),
    ("irfft", lambda: np.fft.irfft(a_complex)),
    ("hfft", lambda: np.fft.hfft(a_complex)),
    ("ihfft", lambda: np.fft.ihfft(a))
]

for name, func in functions_to_test:
    try:
        result = func()
        print(f"{name:10} - Success: {result}")
    except Exception as e:
        print(f"{name:10} - Error: {type(e).__name__}: {e}")