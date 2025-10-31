import numpy as np

print("Testing rfft->irfft roundtrip for different array sizes:")
print("=" * 60)

for size in range(1, 6):
    a = np.ones(size)
    print(f"\nInput size: {size}")
    print(f"Input: {a}")

    rfft_result = np.fft.rfft(a)
    print(f"rfft output size: {rfft_result.shape[0]}")
    print(f"rfft output: {rfft_result}")

    # Calculate what n would be without the parameter
    calculated_n = 2 * (rfft_result.shape[0] - 1)
    print(f"Calculated n = 2*(m-1) = 2*({rfft_result.shape[0]}-1) = {calculated_n}")

    # Try irfft without n
    try:
        result = np.fft.irfft(rfft_result)
        print(f"irfft output size: {result.shape[0]}")
        print(f"irfft output: {result}")
        print(f"Matches original? {np.allclose(a, result)}")
    except ValueError as e:
        print(f"ERROR: {e}")

        # Try with explicit n equal to original size
        result_with_n = np.fft.irfft(rfft_result, n=size)
        print(f"irfft with n={size}: {result_with_n}")
        print(f"Matches original? {np.allclose(a, result_with_n)}")