import numpy as np
import numpy.linalg as la

# Test with the specific failing case
a = np.array([[0.0, 0.0],
              [0.0, 2.2250738585e-313]])

print("Input matrix a:")
print(a)
print(f"a[1,1] = {a[1,1]}")
print(f"Is a[1,1] subnormal? {abs(a[1,1]) < 2.225073858507201e-308 and a[1,1] != 0}")

pinv_a = la.pinv(a)
print("\npinv(a):")
print(pinv_a)
print(f"Contains NaN in pinv: {np.any(np.isnan(pinv_a))}")
print(f"Contains Inf in pinv: {np.any(np.isinf(pinv_a))}")

reconstructed = a @ pinv_a @ a
print("\na @ pinv(a) @ a:")
print(reconstructed)
print("\nExpected (original a):")
print(a)
print(f"\nContains NaN in reconstructed: {np.any(np.isnan(reconstructed))}")
print(f"Are they close? {np.allclose(reconstructed, a, rtol=1e-4, atol=1e-7)}")

# Let's also check the property more carefully
print("\n--- Testing the reconstruction property ---")
try:
    assert np.allclose(reconstructed, a, rtol=1e-4, atol=1e-7), "Reconstruction property failed!"
    print("Test PASSED: a @ pinv(a) @ a ≈ a")
except AssertionError as e:
    print(f"Test FAILED: {e}")
    print("Reconstruction does not match original matrix")