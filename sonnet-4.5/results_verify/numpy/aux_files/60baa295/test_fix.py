import numpy as np

# Test the specific matrix from the bug report
A = np.array([[0.00000000e+000, 2.22507386e-311],
              [2.22507386e-311, 2.22507386e-311]])

print("Testing with proposed fix simulation:")
print(f"Matrix A:")
print(A)

# Calculate SVD
U, s, Vh = np.linalg.svd(A)
print(f"\nSingular values: {s}")

# Current (buggy) tolerance calculation
current_tol = s.max() * max(A.shape) * np.finfo(float).eps
print(f"\nCurrent tolerance calculation:")
print(f"  s.max() * max(A.shape) * eps = {s.max()} * {max(A.shape)} * {np.finfo(float).eps}")
print(f"  = {current_tol}")

# Proposed fix tolerance
proposed_tol = max(s.max() * max(A.shape) * np.finfo(float).eps, np.finfo(float).tiny)
print(f"\nProposed fix tolerance:")
print(f"  max(current_tol, tiny) = max({current_tol}, {np.finfo(float).tiny})")
print(f"  = {proposed_tol}")

# Calculate rank with each tolerance
rank_current = np.sum(s > current_tol)
rank_proposed = np.sum(s > proposed_tol)

print(f"\nRank calculations:")
print(f"  With current tolerance: {rank_current}")
print(f"  With proposed tolerance: {rank_proposed}")

# Verify determinant
det = np.linalg.det(A)
print(f"\nDeterminant: {det}")
print(f"Is singular (det == 0): {det == 0}")
print(f"Expected rank for singular matrix: < {A.shape[0]}")

# Test with slightly larger subnormal values
print("\n" + "="*50)
print("Testing with slightly larger subnormal values:")
B = np.array([[1e-310, 2e-310],
              [3e-310, 6e-310]])

print(f"Matrix B:")
print(B)

det_B = np.linalg.det(B)
rank_B = np.linalg.matrix_rank(B)
print(f"\nDeterminant: {det_B}")
print(f"Determinant == 0: {det_B == 0}")
print(f"matrix_rank: {rank_B}")

U_B, s_B, Vh_B = np.linalg.svd(B)
tol_B = s_B.max() * max(B.shape) * np.finfo(float).eps
print(f"Singular values: {s_B}")
print(f"Default tolerance: {tol_B}")

# Test with normal (non-subnormal) values
print("\n" + "="*50)
print("Testing with normal values (scaled up):")
C = A * 1e308  # Scale to normal range
print(f"Matrix C (scaled A):")
print(C)

det_C = np.linalg.det(C)
rank_C = np.linalg.matrix_rank(C)
print(f"\nDeterminant: {det_C}")
print(f"matrix_rank: {rank_C}")

U_C, s_C, Vh_C = np.linalg.svd(C)
tol_C = s_C.max() * max(C.shape) * np.finfo(float).eps
print(f"Singular values: {s_C}")
print(f"Default tolerance: {tol_C}")
print(f"Singular values > tolerance: {np.sum(s_C > tol_C)}")