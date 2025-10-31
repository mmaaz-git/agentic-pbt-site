import numpy as np
from scipy.signal import tf2ss, ss2tf

# Test case: constant transfer function H(s) = 1
b_orig = np.array([1.0])
a_orig = np.array([1.0])

print("Original transfer function:")
print(f"  Numerator: {b_orig}")
print(f"  Denominator: {a_orig}")
print(f"  This represents H(s) = 1 (a constant)")
print()

# Convert to state-space
A, B, C, D = tf2ss(b_orig, a_orig)
print("State-space representation:")
print(f"  A = {A}")
print(f"  B = {B}")
print(f"  C = {C}")
print(f"  D = {D}")
print()

# Convert back to transfer function
b_result, a_result = ss2tf(A, B, C, D)
print("Reconstructed transfer function:")
print(f"  Numerator: {b_result}")
print(f"  Denominator: {a_result}")
print()

# Analysis
print("Analysis:")
print(f"  Original degree: {len(b_orig) - 1} / {len(a_orig) - 1}")
if b_result.ndim == 2:
    b_result_1d = b_result[0]
else:
    b_result_1d = b_result
print(f"  Reconstructed degree: {len(b_result_1d) - 1} / {len(a_result) - 1}")
print()

print("Issue:")
print("  The original transfer function H(s) = 1 has degree 0.")
print("  After round-trip conversion, we get H(s) = (s + 0)/(s + 0),")
print("  which has degree 1 and a removable singularity at s=0.")
print("  This violates the expected round-trip property.")