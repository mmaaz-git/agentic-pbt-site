import numpy as np
from scipy import signal

original_signal = np.array([451211.0, 0.0, 0.0, 0.0, 0.0, 1.0])
divisor = np.array([1.25, 79299.0])

recorded = signal.convolve(divisor, original_signal)
quotient, remainder = signal.deconvolve(recorded, divisor)
reconstructed = signal.convolve(divisor, quotient) + remainder

print(f"Recorded:      {recorded}")
print(f"Reconstructed: {reconstructed}")
print(f"Max difference: {np.max(np.abs(reconstructed - recorded))}")