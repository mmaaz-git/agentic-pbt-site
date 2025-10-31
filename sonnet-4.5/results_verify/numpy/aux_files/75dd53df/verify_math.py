import numpy as np
import numpy.linalg as la

print("=" * 60)
print("MATHEMATICAL VERIFICATION")
print("=" * 60)

# The matrix in question
a = np.array([[0.00000000e+00, 1.17549435e-38, 0.00000000e+00],
              [1.00000000e+00, 1.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

print("Matrix A:")
print(a)
print()

# Let's manually verify what eigenvector [1, 0, 0] gives us
v_test = np.array([1., 0., 0.])
print(f"Testing v = {v_test}")
result = a @ v_test
print(f"A @ v = {result}")
print(f"For eigenvalue 0, we would need A @ v = 0")
print(f"Clearly, {result} ≠ 0")
print()

# Let's compute eigenvalues using characteristic polynomial
print("Characteristic polynomial det(A - λI) = 0:")
print("For λ = 0:")
det_at_0 = la.det(a)
print(f"det(A) = {det_at_0}")
print("Since det(A) = 0, λ = 0 is indeed an eigenvalue")
print()

# Let's find the actual eigenvector for λ = 0 by solving (A - 0*I)v = 0
print("To find eigenvector for λ = 0, we solve A @ v = 0:")
print("This means we need the null space of A")
print()

# Manual null space calculation
print("The equations A @ v = 0 expand to:")
print("0*v0 + ε*v1 + 0*v2 = 0  =>  ε*v1 = 0")
print("1*v0 + 1*v1 + 0*v2 = 0  =>  v0 + v1 = 0")
print("0*v0 + 0*v1 + 0*v2 = 0  =>  always satisfied")
print()
print(f"Since ε = {1.17549435e-38} ≈ 0 but not exactly 0:")
print("From first equation: v1 = 0")
print("From second equation: v0 + v1 = 0 => v0 = 0")
print("v2 is free")
print("So one eigenvector is [0, 0, 1]")
print()

v_correct = np.array([0., 0., 1.])
print(f"Testing correct eigenvector v = {v_correct}:")
print(f"A @ v = {a @ v_correct}")
print(f"This equals 0*v = {0 * v_correct}")
print()

# Now let's see what numpy is doing with the extreme small value
epsilon = 1.17549435e-38
print(f"The value ε = {epsilon}")
print(f"This is approximately 2^(-126) = {2**(-126)}")
print(f"Float32 smallest normal: {np.finfo(np.float32).tiny}")
print(f"Float64 smallest normal: {np.finfo(np.float64).tiny}")
print()

# Let's see what happens if we treat epsilon as exactly 0
a_zero = np.array([[0., 0., 0.],
                   [1., 1., 0.],
                   [0., 0., 0.]])
print("If we set ε = 0 exactly:")
eigenvalues_zero, eigenvectors_zero = la.eig(a_zero)
print(f"Eigenvalues: {eigenvalues_zero}")
print(f"Eigenvectors:\n{eigenvectors_zero}")

# Verify these
for i in range(3):
    lam = eigenvalues_zero[i]
    v = eigenvectors_zero[:, i]
    lhs = a_zero @ v
    rhs = lam * v
    print(f"Eigenpair {i}: λ={lam}, v={v}")
    print(f"  A@v = {lhs}, λ*v = {rhs}")
    print(f"  Equal? {np.allclose(lhs, rhs)}")