import numpy as np
import traceback

print("Testing numpy.fft irfft family functions with single-element arrays")
print("="*70)

# Test 1: irfft on single-element array
print("\nTest 1: irfft on single-element array")
print("-"*40)
try:
    x = np.array([1.0])
    print(f"Original array: {x}")
    rfft_result = np.fft.rfft(x)
    print(f"rfft result: {rfft_result}")
    print(f"rfft result shape: {rfft_result.shape}")
    print(f"Default n would be: 2*(1-1) = {2*(rfft_result.shape[0]-1)}")
    result = np.fft.irfft(rfft_result)
    print(f"irfft result: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

# Test with workaround
print("\nTest 1 with workaround (n=1):")
try:
    result_workaround = np.fft.irfft(rfft_result, n=1)
    print(f"irfft(rfft_result, n=1) result: {result_workaround}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: irfft2 on single-element 2D array
print("\n\nTest 2: irfft2 on single-element 2D array")
print("-"*40)
try:
    x2d = np.array([[1.0]])
    print(f"Original 2D array: {x2d}")
    rfft2_result = np.fft.rfft2(x2d)
    print(f"rfft2 result: {rfft2_result}")
    print(f"rfft2 result shape: {rfft2_result.shape}")
    result2 = np.fft.irfft2(rfft2_result)
    print(f"irfft2 result: {result2}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

# Test with workaround
print("\nTest 2 with workaround (s=(1,1)):")
try:
    result2_workaround = np.fft.irfft2(rfft2_result, s=(1, 1))
    print(f"irfft2(rfft2_result, s=(1,1)) result: {result2_workaround}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: irfftn on single-element array
print("\n\nTest 3: irfftn on single-element array")
print("-"*40)
try:
    x = np.array([1.0])
    print(f"Original array: {x}")
    rfftn_result = np.fft.rfftn(x)
    print(f"rfftn result: {rfftn_result}")
    print(f"rfftn result shape: {rfftn_result.shape}")
    result3 = np.fft.irfftn(rfftn_result)
    print(f"irfftn result: {result3}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

# Test with workaround
print("\nTest 3 with workaround (s=(1,)):")
try:
    result3_workaround = np.fft.irfftn(rfftn_result, s=(1,))
    print(f"irfftn(rfftn_result, s=(1,)) result: {result3_workaround}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 4: hfft on single-element array
print("\n\nTest 4: hfft on single-element array")
print("-"*40)
try:
    x_hermitian = np.array([1.0+0j])
    print(f"Original hermitian array: {x_hermitian}")
    print(f"Array shape: {x_hermitian.shape}")
    print(f"Default n would be: 2*(1-1) = {2*(x_hermitian.shape[0]-1)}")
    result4 = np.fft.hfft(x_hermitian)
    print(f"hfft result: {result4}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

# Test with workaround
print("\nTest 4 with workaround (n=1):")
try:
    result4_workaround = np.fft.hfft(x_hermitian, n=1)
    print(f"hfft(x_hermitian, n=1) result: {result4_workaround}")
except Exception as e:
    print(f"ERROR: {e}")

# Test round-trip property
print("\n\nTest round-trip property:")
print("-"*40)
print("Testing if irfft(rfft(x)) == x for single element")
x_single = np.array([1.0])
print(f"Original: {x_single}")
rfft_x = np.fft.rfft(x_single)
print(f"After rfft: {rfft_x}")
try:
    irfft_x = np.fft.irfft(rfft_x)
    print(f"After irfft (without n): {irfft_x}")
    print(f"Round-trip successful: {np.allclose(x_single, irfft_x)}")
except Exception as e:
    print(f"Round-trip FAILED without n parameter: {e}")

print("\nUsing n=1:")
irfft_x_with_n = np.fft.irfft(rfft_x, n=1)
print(f"After irfft (with n=1): {irfft_x_with_n}")
print(f"Round-trip successful: {np.allclose(x_single, irfft_x_with_n)}")

# Test with larger arrays to show it works
print("\n\nTest with 2-element array for comparison:")
print("-"*40)
x_two = np.array([1.0, 2.0])
print(f"Original: {x_two}")
rfft_two = np.fft.rfft(x_two)
print(f"After rfft: {rfft_two}, shape: {rfft_two.shape}")
print(f"Default n would be: 2*({rfft_two.shape[0]}-1) = {2*(rfft_two.shape[0]-1)}")
irfft_two = np.fft.irfft(rfft_two)
print(f"After irfft: {irfft_two}")
print(f"Round-trip successful: {np.allclose(x_two, irfft_two)}")