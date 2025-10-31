import numpy as np

# Test 1: Simple 1D size-1 array
print("Test 1: 1D size-1 array")
print("-" * 40)
arr = np.array([5.0])
print(f'Original: {arr}, shape: {arr.shape}')

fft_result = np.fft.rfftn(arr)
print(f'After rfftn: {fft_result}, shape: {fft_result.shape}')

try:
    irfft_result = np.fft.irfftn(fft_result)
    print(f'After irfftn: {irfft_result}')
except ValueError as e:
    print(f'ERROR: {e}')

print()

# Test 2: 2D array with size-1
print("Test 2: 2D array with size-1")
print("-" * 40)
arr2d = np.array([[5.0]])
print(f'Original: {arr2d}, shape: {arr2d.shape}')

fft_result2d = np.fft.rfftn(arr2d)
print(f'After rfftn: {fft_result2d}, shape: {fft_result2d.shape}')

try:
    irfft_result2d = np.fft.irfftn(fft_result2d)
    print(f'After irfftn: {irfft_result2d}')
except ValueError as e:
    print(f'ERROR: {e}')

print()

# Test 3: Direct rfft/irfft test
print("Test 3: Direct rfft/irfft")
print("-" * 40)
arr = np.array([5.0])
print(f'Original: {arr}, shape: {arr.shape}')

rfft_result = np.fft.rfft(arr)
print(f'After rfft: {rfft_result}, shape: {rfft_result.shape}')

try:
    irfft_result = np.fft.irfft(rfft_result)
    print(f'After irfft: {irfft_result}')
except ValueError as e:
    print(f'ERROR: {e}')