import numpy as np
from scipy import signal

signal_arr = np.array([1.0, 2.0, 3.0])
divisor = np.array([0.0, 1.0])

try:
    quotient, remainder = signal.deconvolve(signal_arr, divisor)
    print(f"Success: quotient={quotient}, remainder={remainder}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")