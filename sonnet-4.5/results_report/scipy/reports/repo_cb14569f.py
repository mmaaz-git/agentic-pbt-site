import numpy as np
import scipy.signal

signal = np.array([1.0, 2.0])
divisor = np.array([0.0, 1.0])

quotient, remainder = scipy.signal.deconvolve(signal, divisor)
print(f"Quotient: {quotient}")
print(f"Remainder: {remainder}")