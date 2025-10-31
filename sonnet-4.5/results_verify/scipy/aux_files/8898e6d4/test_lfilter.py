import numpy as np
from scipy import signal

# This replicates what scipy.signal.deconvolve does internally
divisor = np.array([0.3125, 74.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
quotient = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
convolved = signal.convolve(divisor, quotient)

print("REPLICATING SCIPY DECONVOLVE IMPLEMENTATION")
print("=" * 60)

N = len(convolved)
D = len(divisor)

# This is what scipy.signal.deconvolve does internally:
input_signal = np.zeros(N - D + 1, dtype=np.float64)
input_signal[0] = 1
quot_lfilter = signal.lfilter(convolved, divisor, input_signal)
rem_lfilter = convolved - signal.convolve(divisor, quot_lfilter, mode='full')

print(f"Using lfilter (scipy's method):")
print(f"  Quotient from lfilter: {quot_lfilter}")
print(f"  Remainder from lfilter: {rem_lfilter}")

# Reconstruct
reconstructed_lfilter = signal.convolve(divisor, quot_lfilter) + rem_lfilter
print(f"\n  Reconstructed[15]: {reconstructed_lfilter[15]}")
print(f"  Original[15]: {convolved[15]}")
print(f"  Match? {np.allclose(convolved, reconstructed_lfilter)}")

# Compare with direct polynomial division
print(f"\nUsing polydiv (should be correct):")
quot_poly, rem_poly = np.polydiv(convolved, divisor)
reconstructed_poly = np.convolve(divisor, quot_poly) + rem_poly
print(f"  Quotient from polydiv: {quot_poly}")
print(f"  Reconstructed[15]: {reconstructed_poly[15] if len(reconstructed_poly) > 15 else 'N/A'}")
print(f"  Match? {np.allclose(convolved[:len(reconstructed_poly)], reconstructed_poly)}")