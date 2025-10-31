import numpy as np
from scipy import signal

print("Test 1: With signal_arr = [1.0, 2.0, 3.0], divisor = [0.0, 1.0]")
signal_arr = np.array([1.0, 2.0, 3.0])
divisor = np.array([0.0, 1.0])

try:
    quotient, remainder = signal.deconvolve(signal_arr, divisor)
    print(f"Success: quotient={quotient}, remainder={remainder}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\nTest 2: With signal_arr = [0.0, 0.0], divisor = [0.0, 1.0]")
signal_arr = np.array([0.0, 0.0])
divisor = np.array([0.0, 1.0])

try:
    quotient, remainder = signal.deconvolve(signal_arr, divisor)
    print(f"Success: quotient={quotient}, remainder={remainder}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\nTest 3: Control test with valid divisor [1.0, 0.0]")
signal_arr = np.array([1.0, 2.0, 3.0])
divisor = np.array([1.0, 0.0])

try:
    quotient, remainder = signal.deconvolve(signal_arr, divisor)
    print(f"Success: quotient={quotient}, remainder={remainder}")

    # Verify the property
    reconstructed = signal.convolve(divisor, quotient, mode='full') + remainder
    trimmed = reconstructed[:len(signal_arr)]
    print(f"Original: {signal_arr}")
    print(f"Reconstructed: {trimmed}")
    print(f"Match: {np.allclose(trimmed, signal_arr)}")
except ValueError as e:
    print(f"ValueError: {e}")