import numpy as np

# Test some real-world scenarios where small values might appear

print("Real-world scenarios where small values may occur:")
print("="*60)

# 1. Numerical differentiation residuals
print("\n1. Numerical differentiation (machine epsilon scale):")
epsilon = np.finfo(float).eps
A = np.array([[epsilon, epsilon], [1.0, 1.0]])
eigenvalues, eigenvectors = np.linalg.eig(A)
v0 = eigenvectors[:, 0]
error = np.max(np.abs(A @ v0 - eigenvalues[0] * v0))
print(f"  Epsilon: {epsilon:.2e}")
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Error: {error:.2e}")
print(f"  Works correctly: {error < 1e-9}")

# 2. Quantum physics - very small probabilities
print("\n2. Quantum physics (very small probability amplitudes):")
small_prob = 1e-60  # Small quantum probability
A = np.array([[small_prob, small_prob], [1.0, 1.0]])
eigenvalues, eigenvectors = np.linalg.eig(A)
v0 = eigenvectors[:, 0]
error = np.max(np.abs(A @ v0 - eigenvalues[0] * v0))
print(f"  Small probability: {small_prob:.2e}")
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Error: {error:.2e}")
print(f"  Works correctly: {error < 1e-9}")

# 3. Astrophysics - ratios of particle masses
print("\n3. Astrophysics (neutrino to solar mass ratio ~1e-60):")
ratio = 1e-60
A = np.array([[ratio, ratio], [1.0, 1.0]])
eigenvalues, eigenvectors = np.linalg.eig(A)
v0 = eigenvectors[:, 0]
error = np.max(np.abs(A @ v0 - eigenvalues[0] * v0))
print(f"  Mass ratio: {ratio:.2e}")
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Error: {error:.2e}")
print(f"  Works correctly: {error < 1e-9}")

# 4. Machine learning - gradient descent near convergence
print("\n4. Machine learning (very small gradients):")
tiny_gradient = 1e-100
A = np.array([[tiny_gradient, tiny_gradient], [1.0, 1.0]])
eigenvalues, eigenvectors = np.linalg.eig(A)
v0 = eigenvectors[:, 0]
error = np.max(np.abs(A @ v0 - eigenvalues[0] * v0))
print(f"  Gradient magnitude: {tiny_gradient:.2e}")
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Error: {error:.2e}")
print(f"  Works correctly: {error < 1e-9}")

# 5. Statistical physics - Boltzmann factors at very low temperatures
print("\n5. Statistical physics (Boltzmann factor at extremely low temperature):")
boltzmann_factor = np.exp(-1000)  # exp(-E/kT) for high E or low T
print(f"  Boltzmann factor: {boltzmann_factor:.2e}")
A = np.array([[boltzmann_factor, boltzmann_factor], [1.0, 1.0]])
eigenvalues, eigenvectors = np.linalg.eig(A)
v0 = eigenvectors[:, 0]
error = np.max(np.abs(A @ v0 - eigenvalues[0] * v0))
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Error: {error:.2e}")
print(f"  Works correctly: {error < 1e-9}")