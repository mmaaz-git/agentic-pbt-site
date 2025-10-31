import numpy as np
from scipy import signal

# Test case that fails
signal_arr = np.array([1.0, 2.0, 3.0])
divisor_with_zero = np.array([0.0, 1.0])

print("Testing with leading zero divisor [0.0, 1.0]:")
try:
    quotient, remainder = signal.deconvolve(signal_arr, divisor_with_zero)
    print(f"Success: quotient={quotient}, remainder={remainder}")
except Exception as e:
    print(f"Failed: {e}")

# Workaround: strip leading zeros
divisor_stripped = np.array([1.0])
print("\nTesting with stripped divisor [1.0]:")
try:
    quotient, remainder = signal.deconvolve(signal_arr, divisor_stripped)
    print(f"Success: quotient={quotient}, remainder={remainder}")

    # Verify the mathematical property holds
    reconstructed = signal.convolve(divisor_stripped, quotient, mode='full') + remainder
    print(f"Original signal: {signal_arr}")
    print(f"Reconstructed: {reconstructed[:len(signal_arr)]}")
    print(f"Match: {np.allclose(reconstructed[:len(signal_arr)], signal_arr)}")
except Exception as e:
    print(f"Failed: {e}")

# Test that [0, 1] and [1] are mathematically equivalent as polynomial divisors
print("\nMathematical equivalence test:")
print(f"Polynomial [0, 1] evaluated at x=2: {0*2 + 1} = 1")
print(f"Polynomial [1] evaluated at x=2: {1} = 1")
print("These represent the same polynomial (constant 1)")