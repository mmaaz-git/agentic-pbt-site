import numpy as np

eps = np.finfo(float).eps
print(f"Machine epsilon (eps) = {eps}")
print(f"100 * eps = {100 * eps}")
print(f"In scientific notation: {100 * eps:.2e}")

# Check if this matches the value we tested
print(f"\nThis matches the tested value: {100*eps == 2.220446049250313e-14}")