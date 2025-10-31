import numpy as np
import scipy.fft

print("Testing minimal reproduction case...")
x = np.array([5.0])
print(f"Input array x: {x}")
print(f"Input shape: {x.shape}")

rfft_out = scipy.fft.rfft(x)
print(f"rfft output: {rfft_out}")
print(f"rfft output shape: {rfft_out.shape}")

try:
    irfft_out = scipy.fft.irfft(rfft_out)
    print(f"irfft output: {irfft_out}")
except ValueError as e:
    print(f"ValueError raised: {e}")

# Let's also test with explicit n parameter
print("\nTesting with explicit n=1 parameter...")
try:
    irfft_out = scipy.fft.irfft(rfft_out, n=1)
    print(f"irfft output with n=1: {irfft_out}")
    print(f"Does it match original? {np.allclose(irfft_out, x)}")
except Exception as e:
    print(f"Error with n=1: {e}")

# Test with larger arrays to see the pattern
print("\nTesting with 2-element array...")
x2 = np.array([5.0, 3.0])
rfft_out2 = scipy.fft.rfft(x2)
print(f"rfft([5.0, 3.0]) shape: {rfft_out2.shape}")
irfft_out2 = scipy.fft.irfft(rfft_out2)
print(f"irfft works, output shape: {irfft_out2.shape}")
print(f"Round trip successful? {np.allclose(irfft_out2, x2)}")

print("\nTesting with 3-element array...")
x3 = np.array([5.0, 3.0, 2.0])
rfft_out3 = scipy.fft.rfft(x3)
print(f"rfft([5.0, 3.0, 2.0]) shape: {rfft_out3.shape}")
try:
    irfft_out3 = scipy.fft.irfft(rfft_out3)
    print(f"irfft output shape: {irfft_out3.shape}")
    print(f"Round trip successful? {np.allclose(irfft_out3, x3)}")
except Exception as e:
    print(f"Error: {e}")
    # Try with explicit n
    irfft_out3 = scipy.fft.irfft(rfft_out3, n=3)
    print(f"With n=3, output: {irfft_out3}")
    print(f"Round trip successful? {np.allclose(irfft_out3, x3)}")