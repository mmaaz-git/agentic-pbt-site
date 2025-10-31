import numpy as np
import scipy.special as special

# Test what happens with the Bessel function i0 for large values
test_values = [100, 300, 500, 700, 710, 750, 800, 1000]

for beta in test_values:
    try:
        result = special.i0(beta)
        print(f"i0({beta}) = {result}")
    except Exception as e:
        print(f"i0({beta}) failed with: {e}")

# Also test the division that happens in kaiser
print("\n--- Testing division scenario ---")
beta = 710.0
numerator = special.i0(0.0)  # This would be the edge case at n=0 or n=M-1
denominator = special.i0(beta)
print(f"i0(0.0) = {numerator}")
print(f"i0({beta}) = {denominator}")
print(f"i0(0.0) / i0({beta}) = {numerator / denominator}")