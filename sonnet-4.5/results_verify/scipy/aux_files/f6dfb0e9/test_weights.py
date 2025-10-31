import numpy as np

# Simulate what happens in _derivative_weights with step_factor=1.0
n = 4  # half of order 8
fac = 1.0  # This is the problem value

# Central difference weights calculation from lines 674-682
i = np.arange(-n, n + 1)
p = np.abs(i) - 1.
s = np.sign(i)

h = s / fac ** p
print("h values for central difference:")
print(h)
print("Note the duplicate values at indices 3,4 and 5,6:", h[[3, 4]], h[[5, 6]])

print("\nConstructing Vandermonde matrix...")
try:
    A = np.vander(h, increasing=True).T
    print("Matrix shape:", A.shape)
    print("Matrix determinant:", np.linalg.det(A))

    b = np.zeros(2*n + 1)
    b[1] = 1
    weights = np.linalg.solve(A, b)
    print("Weights computed successfully:", weights)
except np.linalg.LinAlgError as e:
    print(f"LinAlgError: {e}")
    print("This is what causes the crash in scipy.differentiate.derivative")