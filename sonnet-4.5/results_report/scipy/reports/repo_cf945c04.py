import scipy.special as sp
import numpy as np

# Test cases showing NaN returns for small delta values
print("Testing scipy.special.pseudo_huber with small delta values:")
print("=" * 60)

# Test case from the bug report
delta = 1e-200
r = 1.0
result = sp.pseudo_huber(delta, r)
print(f"sp.pseudo_huber({delta}, {r}) = {result}")

# Additional test cases to show the pattern
test_cases = [
    (1e-100, 1.0),
    (1e-150, 1.0),
    (1e-190, 1.0),
    (1e-200, 1.0),
    (1e-250, 1.0),
    (1e-300, 1.0),
    (2.3581411596114265e-203, 1.0)  # Example from hypothesis test
]

print("\nAdditional test cases:")
for delta, r in test_cases:
    result = sp.pseudo_huber(delta, r)
    print(f"sp.pseudo_huber({delta:.2e}, {r}) = {result}")

# Show expected value calculation
print("\n" + "=" * 60)
print("Expected behavior:")
print("For small delta and r=1.0, pseudo_huber should return approximately |r| - delta ≈ 1.0")
print("The mathematical formula δ²(√(1 + (r/δ)²) - 1) is well-defined for all positive delta.")