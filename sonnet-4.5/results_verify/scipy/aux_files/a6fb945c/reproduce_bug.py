import scipy.sparse as sp

print("Testing k=n case (should work according to bug report):")
E1 = sp.eye_array(3, k=3, format='csr')
print(f"eye_array(3, k=3): nnz={E1.nnz}")

print("\nTesting k=n+1 case (should fail according to bug report):")
try:
    E2 = sp.eye_array(3, k=4, format='csr')
    print(f"eye_array(3, k=4): nnz={E2.nnz}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\nAdditional tests:")
print("Testing k=-3 (negative boundary):")
E3 = sp.eye_array(3, k=-3, format='csr')
print(f"eye_array(3, k=-3): nnz={E3.nnz}")

print("\nTesting k=-4 (negative out of bounds):")
try:
    E4 = sp.eye_array(3, k=-4, format='csr')
    print(f"eye_array(3, k=-4): nnz={E4.nnz}")
except ValueError as e:
    print(f"ValueError: {e}")