import numpy as np
import scipy.signal

# Test case from bug report
signal = np.array([1.0, 1.0])
divisor = np.array([1.1754943508222875e-38, 1.0])

print("Testing deconvolve with small leading coefficient:")
print(f"Signal:  {signal}")
print(f"Divisor: {divisor}")
print(f"Divisor[0]: {divisor[0]}")

quotient, remainder = scipy.signal.deconvolve(signal, divisor)

print(f"\nQuotient:  {quotient}")
print(f"Remainder: {remainder}")

reconstructed = scipy.signal.convolve(divisor, quotient) + remainder

print(f"\nOriginal signal:  {signal}")
print(f"Reconstructed:    {reconstructed}")
print(f"Difference:       {reconstructed - signal}")

# Check if round-trip property holds
if np.allclose(reconstructed, signal, rtol=1e-10, atol=1e-10):
    print("\n✓ Round-trip property HOLDS")
else:
    print("\n✗ Round-trip property VIOLATED")

# Also test what happens when we divide by divisor[0]
print(f"\n1.0 / divisor[0] = {1.0 / divisor[0]}")
print(f"Is this inf? {np.isinf(1.0 / divisor[0])}")