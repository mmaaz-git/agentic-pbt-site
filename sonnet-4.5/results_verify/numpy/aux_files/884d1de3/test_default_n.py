import numpy as np

print("Testing with and without specifying 'n' parameter:\n")

# Test case 1: Size-1 array
arr = np.array([5.0])
print(f"Array: {arr}")

# With explicit n parameter (should work according to docs)
fft_result = np.fft.rfft(arr)
print(f"FFT result: {fft_result}")

print("\nWith explicit n=1:")
try:
    result_with_n = np.fft.irfft(fft_result, n=1)
    print(f"  Success: {result_with_n}")
except Exception as e:
    print(f"  Failed: {e}")

print("\nWithout n (default behavior):")
try:
    result_without_n = np.fft.irfft(fft_result)
    print(f"  Success: {result_without_n}")
except Exception as e:
    print(f"  Failed: {e}")

print("\n" + "="*50)
print("Checking how n is calculated for size-1:")
print(f"  FFT result shape: {fft_result.shape}")
print(f"  According to code: n = (shape[axis] - 1) * 2")
print(f"  Calculated n = ({fft_result.shape[0]} - 1) * 2 = {(fft_result.shape[0] - 1) * 2}")
print(f"  This gives n=0, which causes the error!")

print("\n" + "="*50)
print("Testing with size-2 array for comparison:")
arr2 = np.array([5.0, 3.0])
fft_result2 = np.fft.rfft(arr2)
print(f"  Array: {arr2}")
print(f"  FFT result: {fft_result2}, shape: {fft_result2.shape}")
print(f"  Calculated n = ({fft_result2.shape[0]} - 1) * 2 = {(fft_result2.shape[0] - 1) * 2}")
try:
    result2 = np.fft.irfft(fft_result2)
    print(f"  irfft without n: Success - {result2}")
except Exception as e:
    print(f"  irfft without n: Failed - {e}")