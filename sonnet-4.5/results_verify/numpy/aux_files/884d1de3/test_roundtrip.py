import numpy as np

# Test the expected round-trip property from documentation
print("Testing round-trip property for different sizes:\n")

for size in [1, 2, 3, 4, 5]:
    arr = np.random.rand(size)
    print(f"Size {size}: ", end="")
    try:
        # According to docs: irfft(rfft(a), len(a)) == a
        fft_result = np.fft.rfft(arr)
        reconstructed = np.fft.irfft(fft_result, len(arr))

        # Check if round-trip works
        if np.allclose(reconstructed, arr):
            print("✓ Round-trip successful")
        else:
            print("✗ Round-trip failed (values don't match)")
    except Exception as e:
        print(f"✗ Round-trip failed with error: {e}")

print("\nTesting N-dimensional round-trip:\n")

for shape in [(1,), (1, 1), (2, 1), (1, 2), (2, 2)]:
    arr = np.random.rand(*shape)
    print(f"Shape {shape}: ", end="")
    try:
        # According to docs: irfftn(rfftn(a), a.shape) == a
        fft_result = np.fft.rfftn(arr)
        reconstructed = np.fft.irfftn(fft_result, arr.shape)

        # Check if round-trip works
        if np.allclose(reconstructed, arr):
            print("✓ Round-trip successful")
        else:
            print("✗ Round-trip failed (values don't match)")
    except Exception as e:
        print(f"✗ Round-trip failed with error: {e}")