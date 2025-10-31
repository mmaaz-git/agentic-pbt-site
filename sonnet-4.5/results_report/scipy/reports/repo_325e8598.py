import numpy as np
import scipy.signal.windows as w

# Test with various M values
for M in [2, 5, 10, 100, 1000]:
    window = w.blackman(M)
    min_val = window.min()
    has_negative = np.any(window < 0)
    negative_count = np.sum(window < 0)

    print(f"M={M}:")
    print(f"  min value = {min_val:.20e}")
    print(f"  has negative values = {has_negative}")
    print(f"  number of negative values = {negative_count}")
    if has_negative:
        negative_indices = np.where(window < 0)[0]
        print(f"  negative value indices = {negative_indices.tolist()}")
        print(f"  negative values = {[f'{window[i]:.20e}' for i in negative_indices]}")
    print(f"  endpoints = [{window[0]:.20e}, {window[-1]:.20e}]")
    print()

# Verify the mathematical expectation
print("Mathematical analysis:")
a0, a1, a2 = 0.42, 0.50, 0.08
theoretical_endpoint = a0 - a1*1.0 + a2*1.0  # cos(0) = 1 for both terms
print(f"Theoretical endpoint value: {theoretical_endpoint}")
print(f"This should be exactly 0.0")
print()

# Check what the actual implementation does
print("What the implementation computes at endpoints:")
M = 10
n = 0  # First endpoint
term1 = 0.42
term2 = -0.50 * np.cos(0)  # cos(0) = 1
term3 = 0.08 * np.cos(0)   # cos(0) = 1
endpoint_value = term1 + term2 + term3
print(f"At n=0: 0.42 - 0.50*cos(0) + 0.08*cos(0) = {term1} + {term2} + {term3} = {endpoint_value}")
print(f"Due to floating point representation: {0.42 - 0.50 + 0.08}")