import numpy as np
import numpy.random as npr

rng = npr.default_rng(42)

# Test with all zeros
alpha_zeros = [0.0, 0.0, 0.0]
result = rng.dirichlet(alpha_zeros)

print(f"Alpha: {alpha_zeros}")
print(f"Result: {result}")
print(f"Sum: {result.sum()}")
print(f"Expected sum: 1.0")
print(f"Violation: Sum is {result.sum()} instead of 1.0")