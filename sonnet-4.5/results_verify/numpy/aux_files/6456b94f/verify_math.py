import numpy as np

# The matrix
a = np.array([[0.00000000e+00, 1.17549435e-38, 0.00000000e+00],
              [1.00000000e+00, 1.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

print("Matrix A:")
print(a)
print()

# The claimed eigenvector for eigenvalue 0
v = np.array([1., 0., 0.])
lam = 0.0

print("Testing eigenvector [1, 0, 0] for eigenvalue 0:")
print(f"v = {v}")
print(f"λ = {lam}")
print()

# Manual computation of A @ v
av_manual = np.zeros(3)
for i in range(3):
    for j in range(3):
        av_manual[i] += a[i, j] * v[j]

print("Manual computation of A @ v:")
print(f"  Row 0: {a[0,0]} * {v[0]} + {a[0,1]} * {v[1]} + {a[0,2]} * {v[2]} = {a[0,0]*v[0] + a[0,1]*v[1] + a[0,2]*v[2]}")
print(f"  Row 1: {a[1,0]} * {v[0]} + {a[1,1]} * {v[1]} + {a[1,2]} * {v[2]} = {a[1,0]*v[0] + a[1,1]*v[1] + a[1,2]*v[2]}")
print(f"  Row 2: {a[2,0]} * {v[0]} + {a[2,1]} * {v[1]} + {a[2,2]} * {v[2]} = {a[2,0]*v[0] + a[2,1]*v[1] + a[2,2]*v[2]}")
print(f"A @ v = {av_manual}")
print()

# Lambda * v
lambdav = lam * v
print(f"λ * v = {lam} * {v} = {lambdav}")
print()

# Check if they're equal
print("Comparison:")
print(f"A @ v     = {av_manual}")
print(f"λ * v     = {lambdav}")
print(f"Different = {av_manual - lambdav}")
print(f"Are they equal? {np.allclose(av_manual, lambdav)}")
print()

# Check what the correct eigenvector for eigenvalue 0 should be
print("Finding null space (eigenspace for λ=0):")
print("We need to solve (A - 0*I) @ v = 0, i.e., A @ v = 0")
print()

# The matrix A is:
# [[0, ε, 0],
#  [1, 1, 0],
#  [0, 0, 0]]
# where ε ≈ 1.17e-38

# For A @ v = 0:
# Row 0: 0*v[0] + ε*v[1] + 0*v[2] = 0 => v[1] = 0 (since ε ≠ 0)
# Row 1: 1*v[0] + 1*v[1] + 0*v[2] = 0 => v[0] + v[1] = 0 => v[0] = 0 (since v[1] = 0)
# Row 2: 0*v[0] + 0*v[1] + 0*v[2] = 0 => always satisfied

print("From row 0: ε*v[1] = 0, and since ε ≠ 0, we need v[1] = 0")
print("From row 1: v[0] + v[1] = 0, and since v[1] = 0, we need v[0] = 0")
print("From row 2: always satisfied")
print()
print("Therefore, the null space consists of vectors [0, 0, c] for any c")
print("The only eigenvector for eigenvalue 0 should be proportional to [0, 0, 1]")
print()

# Test the correct eigenvector
v_correct = np.array([0., 0., 1.])
av_correct = a @ v_correct
lambdav_correct = 0 * v_correct

print("Testing correct eigenvector [0, 0, 1] for eigenvalue 0:")
print(f"A @ v = {av_correct}")
print(f"λ * v = {lambdav_correct}")
print(f"Equal? {np.allclose(av_correct, lambdav_correct)}")