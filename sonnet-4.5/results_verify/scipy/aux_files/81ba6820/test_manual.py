import scipy.fftpack
import numpy as np

result = scipy.fftpack.next_fast_len(0)
print(f"next_fast_len(0) = {result}")

try:
    scipy.fftpack.next_fast_len(-1)
except ValueError as e:
    print(f"next_fast_len(-1) raises: {e}")

try:
    x = np.array([1., 2., 3.])
    scipy.fftpack.fft(x, n=result)
except ValueError as e:
    print(f"Using result in fft: {e}")