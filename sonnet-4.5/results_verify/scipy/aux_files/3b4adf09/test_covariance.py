import numpy as np

# Test if the covariance matrix is positive definite for a 5x5 random matrix
np.random.seed(0)
data = np.random.randn(5, 5)

print(f"Data shape: {data.shape}")
print(f"data.shape[1] > data.shape[0]: {data.shape[1] > data.shape[0]}")
print(f"data.shape[1] == data.shape[0]: {data.shape[1] == data.shape[0]}")

# Calculate covariance matrix
cov = np.cov(data.T)
print(f"\nCovariance matrix shape: {cov.shape}")

# Check eigenvalues to see if positive definite
eigenvalues = np.linalg.eigvals(cov)
print(f"Eigenvalues: {eigenvalues}")
print(f"All eigenvalues positive? {np.all(eigenvalues > 0)}")
print(f"Matrix rank: {np.linalg.matrix_rank(cov)}")
print(f"Matrix condition number: {np.linalg.cond(cov)}")

# Try Cholesky decomposition
try:
    chol = np.linalg.cholesky(cov)
    print(f"\nCholesky decomposition succeeded")
except np.linalg.LinAlgError as e:
    print(f"\nCholesky decomposition failed: {e}")

# Explain the issue
print("\nAnalysis:")
print("When n_obs == n_features, the covariance matrix can be rank-deficient.")
print(f"Here we have {data.shape[0]} observations and {data.shape[1]} features.")
print(f"The covariance matrix has rank {np.linalg.matrix_rank(cov)} out of {cov.shape[0]} possible.")
print("A rank-deficient covariance matrix is not positive definite, causing Cholesky to fail.")