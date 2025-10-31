import numpy as np
import numpy.linalg as la

# The failing matrix from the bug report
a = np.array([[0.00000000e+00, 1.17549435e-38, 0.00000000e+00],
              [1.00000000e+00, 1.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

print("Matrix A:")
print(a)
print()

# Compute eigenvalues and eigenvectors
result = la.eig(a)

# Note: The bug report uses result.eigenvalues and result.eigenvectors
# but numpy.linalg.eig returns a tuple, not a namedtuple with those attributes
# Let's check what it actually returns
print(f"Type of result: {type(result)}")
print(f"Result attributes: {dir(result) if hasattr(result, '__dir__') else 'N/A'}")

# Extract eigenvalues and eigenvectors properly
if hasattr(result, 'eigenvalues'):
    eigenvalues = result.eigenvalues
    eigenvectors = result.eigenvectors
else:
    # Standard numpy returns a tuple (eigenvalues, eigenvectors)
    eigenvalues, eigenvectors = result

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")
print()

# Check the eigenvalue equation for each eigenpair
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]

    # Compute A @ v and lambda * v
    av = a @ v
    lambdav = lam * v

    print(f"Eigenpair {i}:")
    print(f"  Eigenvalue: {lam}")
    print(f"  Eigenvector: {v}")
    print(f"  A @ v = {av}")
    print(f"  lambda * v = {lambdav}")
    print(f"  Difference: {av - lambdav}")
    print(f"  Error norm: {np.linalg.norm(av - lambdav)}")
    print(f"  Equal (allclose)? {np.allclose(av, lambdav, rtol=1e-4, atol=1e-7)}")
    print()