import numpy as np

A = np.array([[     0. ,      0. , 416614.5],
              [416614.5, 416614.5, 416614.5],
              [416614.5, 416614.5, 416614.5]])

det_A = np.linalg.det(A)
det_AT = np.linalg.det(A.T)

print(f"det(A)   = {det_A}")
print(f"det(A.T) = {det_AT}")
print(f"|det(A)|   = {abs(det_A)}")
print(f"|det(A.T)| = {abs(det_AT)}")
print(f"Signs match? {det_A * det_AT > 0}")

print("\nAlternative smaller example from test:")
A2 = np.array([[    0. ,     0. , 27432.5],
               [27432.5, 27432.5, 27432.5],
               [27432.5, 27432.5, 27432.5]])

det_A2 = np.linalg.det(A2)
det_AT2 = np.linalg.det(A2.T)

print(f"\ndet(A2)   = {det_A2}")
print(f"det(A2.T) = {det_AT2}")
print(f"|det(A2)|   = {abs(det_A2)}")
print(f"|det(A2.T)| = {abs(det_AT2)}")
print(f"Signs match? {det_A2 * det_AT2 > 0}")

print("\nMatrix rank and condition number:")
print(f"rank(A) = {np.linalg.matrix_rank(A)}")
print(f"rank(A.T) = {np.linalg.matrix_rank(A.T)}")
print(f"cond(A) = {np.linalg.cond(A)}")

print("\nMathematically, this matrix should have determinant = 0")
print("Rows 2 and 3 are identical, so the matrix is singular")