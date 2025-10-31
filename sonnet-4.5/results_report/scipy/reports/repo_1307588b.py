import numpy as np
import scipy.signal

signal = np.array([65.0] + [0.0]*21 + [1.875])
divisor = np.array([0.125, 0.0, 2.0])

quotient, remainder = scipy.signal.deconvolve(signal, divisor)
reconstructed = scipy.signal.convolve(divisor, quotient, mode='full')
reconstructed[:len(remainder)] += remainder

print(f"Signal length: {len(signal)}")
print(f"Signal: {signal}")
print(f"Divisor: {divisor}")
print(f"Quotient length: {len(quotient)}")
print(f"Remainder length: {len(remainder)}")
print(f"Reconstructed length: {len(reconstructed)}")
print()
print(f"Expected last value: {signal[-1]}")
print(f"Reconstructed last value: {reconstructed[len(signal)-1]}")
print(f"Error: {abs(signal[-1] - reconstructed[len(signal)-1])}")
print(f"Relative error: {abs(signal[-1] - reconstructed[len(signal)-1])/abs(signal[-1])*100:.2f}%")
print(f"Max quotient magnitude: {np.max(np.abs(quotient)):.2e}")
print()
print("Full reconstruction comparison:")
print(f"Original signal: {signal}")
print(f"Reconstructed (trimmed): {reconstructed[:len(signal)]}")
print(f"Difference: {signal - reconstructed[:len(signal)]}")
print(f"Max absolute error: {np.max(np.abs(signal - reconstructed[:len(signal)]))}")