import scipy.fftpack
import numpy as np

# Test with 0 - should raise ValueError according to documentation
print("Testing scipy.fftpack.next_fast_len(0):")
result = scipy.fftpack.next_fast_len(0)
print(f"next_fast_len(0) = {result}")
print(f"Type of result: {type(result)}")
print()

# Test with -1 - should raise ValueError
print("Testing scipy.fftpack.next_fast_len(-1):")
try:
    scipy.fftpack.next_fast_len(-1)
    print("next_fast_len(-1) did not raise an error")
except ValueError as e:
    print(f"next_fast_len(-1) raises ValueError: {e}")
print()

# Test using the result in fft
print("Testing using result=0 in scipy.fftpack.fft:")
try:
    x = np.array([1., 2., 3.])
    fft_result = scipy.fftpack.fft(x, n=result)
    print(f"fft with n={result} succeeded")
except ValueError as e:
    print(f"fft with n={result} raises ValueError: {e}")
print()

# Additional tests to show expected behavior
print("Testing with positive values:")
for val in [1, 2, 5, 10, 100]:
    fast_len = scipy.fftpack.next_fast_len(val)
    print(f"next_fast_len({val}) = {fast_len}")