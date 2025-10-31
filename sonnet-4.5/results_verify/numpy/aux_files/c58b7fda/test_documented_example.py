import numpy as np

print("Testing the exact example from numpy documentation")
print("=" * 50)

# From the hfft documentation example:
signal = np.array([[1, 1.j], [-1.j, 2]])
print(f"Signal from docs: \n{signal}")
print(f"Check Hermitian symmetry: np.conj(signal.T) - signal")
print(np.conj(signal.T) - signal)
print("(Should be close to zero for Hermitian matrix)")

freq_spectrum = np.fft.hfft(signal)
print(f"\nhfft result:\n{freq_spectrum}")

print("\n" + "=" * 50)
print("Let me work through the 1D case from documentation more carefully:")

# The example in the docs shows that for a real signal [1,2,3,4,3,2]
# The FFT is [15+0j, -4+0j, 0+0j, -1+0j, 0+0j, -4+0j]
# And hfft of the first 4 elements should give back the signal

# But what does "first 4 elements" mean for hfft input?
# hfft expects Hermitian TIME domain input

print("\nLet's trace through what the docs mean:")
print("1. Real signal: [1, 2, 3, 4, 3, 2]")
print("2. FFT gives Hermitian frequency domain")
print("3. hfft is NOT for this use case!")
print("4. hfft is for: Hermitian TIME -> Real FREQUENCY")

print("\n" + "=" * 50)
print("Testing simpler cases to understand the pattern:")

# Simple test with minimal Hermitian array
test1 = np.array([1+0j])  # Length 1, always Hermitian
print(f"\nTest with length-1 array: {test1}")
n = 2 * len(test1) - 2  # n = 0, this doesn't make sense!
n = 2 * len(test1) - 1  # Try odd case: n = 1
hfft_out = np.fft.hfft(test1, n)
print(f"hfft(..., n={n}): {hfft_out}")
recovered = np.fft.ihfft(hfft_out)
print(f"ihfft result: {recovered}")
print(f"Match? {np.allclose(recovered, test1)}")

# Test with length 2 Hermitian
test2 = np.array([1+2j, 1-2j])  # Hermitian
print(f"\nTest with length-2 Hermitian: {test2}")
n = 2 * len(test2) - 2  # n = 2
hfft_out = np.fft.hfft(test2, n)
print(f"hfft(..., n={n}): {hfft_out}")
recovered = np.fft.ihfft(hfft_out)
print(f"ihfft result: {recovered}")
print(f"Original: {test2}")
print(f"Match? {np.allclose(recovered, test2)}")

print("\n" + "=" * 50)
print("Key insight: Looking at recovered vs original:")
print("The imaginary parts are getting conjugated!")
print("ihfft(hfft(a)) gives conj(a), not a!")