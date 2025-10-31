import numpy as np
import scipy.fftpack as fftpack

def investigate_signal(x, description):
    print(f"\nInvestigating: {description}")
    print(f"Signal x: {x}")
    print(f"Sum: {np.sum(x)}")

    # Get Fourier transform
    fft_x = np.fft.fft(x)
    print(f"FFT of x: {fft_x}")

    # Apply diff order=1
    diff_x = fftpack.diff(x, order=1)
    print(f"diff(x, 1): {diff_x}")

    fft_diff = np.fft.fft(diff_x)
    print(f"FFT of diff(x): {fft_diff}")

    # Apply diff order=-1 (integration)
    roundtrip = fftpack.diff(diff_x, order=-1)
    print(f"diff(diff(x, 1), -1): {roundtrip}")

    fft_roundtrip = np.fft.fft(roundtrip)
    print(f"FFT of roundtrip: {fft_roundtrip}")

    print(f"Round-trip successful: {np.allclose(roundtrip, x)}")

    # Check DC component
    print(f"DC component of x (fft[0]): {fft_x[0]}")
    print(f"DC component of diff(x): {fft_diff[0]}")
    print(f"DC component of roundtrip: {fft_roundtrip[0]}")

    return np.allclose(roundtrip, x)

# Test cases
investigate_signal(np.array([-0.5, 0.5]), "[-0.5, 0.5] (FAILS)")
investigate_signal(np.array([-1, 0, 1]), "[-1, 0, 1] (WORKS)")
investigate_signal(np.array([1, -1, 1, -1]), "[1, -1, 1, -1] (FAILS)")
investigate_signal(np.array([0, 1, 0, -1]), "[0, 1, 0, -1] (WORKS)")

# Additional test to understand the pattern
print("\n" + "="*50)
print("Understanding the kernel function:")
print("="*50)

# The diff function uses kernel(k) = (c*k)^order for k!=0, 0 for k==0
# For order=1, kernel(k) = k for k!=0
# For order=-1, kernel(k) = 1/k for k!=0

# When we have even length and odd order, Nyquist mode is taken as zero
x = np.array([-0.5, 0.5])
n = len(x)
print(f"\nFor x = {x} (length {n}):")
print("Fourier frequencies: k = 0, 1")
print("For diff with order=1: kernel = [0, 1]")
print("For diff with order=-1: kernel = [0, 1/1] = [0, 1]")
print("But since order=1 is odd and len=2 is even, Nyquist mode is zeroed")
print("Nyquist frequency for n=2 is k=1")
print("So the kernel for order=1 becomes [0, 0] for the Nyquist mode")