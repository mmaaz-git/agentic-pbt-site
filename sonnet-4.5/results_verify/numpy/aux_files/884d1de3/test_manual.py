import numpy as np

print("Test 1: 1D array with single element")
print("="*50)
arr = np.array([5.0])
print(f'Original: {arr}, shape: {arr.shape}')

fft_result = np.fft.rfftn(arr)
print(f'After rfftn: {fft_result}, shape: {fft_result.shape}')

try:
    irfft_result = np.fft.irfftn(fft_result)
    print(f'After irfftn: {irfft_result}')
except ValueError as e:
    print(f'ERROR: {e}')

print("\nTest 2: 2D array with single element")
print("="*50)
arr2d = np.array([[5.0]])
print(f'Original: {arr2d}, shape: {arr2d.shape}')

fft_result2d = np.fft.rfftn(arr2d)
print(f'After rfftn: {fft_result2d}, shape: {fft_result2d.shape}')

try:
    irfft_result2d = np.fft.irfftn(fft_result2d)
    print(f'After irfftn: {irfft_result2d}')
except ValueError as e:
    print(f'ERROR: {e}')

print("\nTest 3: Testing irfft directly with size-1")
print("="*50)
arr = np.array([5.0])
print(f'Original: {arr}, shape: {arr.shape}')

fft_result = np.fft.rfft(arr)
print(f'After rfft: {fft_result}, shape: {fft_result.shape}')

try:
    irfft_result = np.fft.irfft(fft_result)
    print(f'After irfft: {irfft_result}')
except ValueError as e:
    print(f'ERROR: {e}')