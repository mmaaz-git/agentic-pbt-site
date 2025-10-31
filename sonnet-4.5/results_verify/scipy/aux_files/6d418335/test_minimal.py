import numpy as np
import scipy.signal as signal

signal_arr = np.array([0.0, 0.0])
divisor_arr = np.array([0.0, 1.0])

print("Testing with signal_array=[0.0, 0.0], divisor_array=[0.0, 1.0]")
try:
    quotient, remainder = signal.deconvolve(signal_arr, divisor_arr)
    print(f"Success! quotient={quotient}, remainder={remainder}")
except ValueError as e:
    print(f"ValueError: {e}")