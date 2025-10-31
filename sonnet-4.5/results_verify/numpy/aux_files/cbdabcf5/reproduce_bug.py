import numpy as np

print("=== Reproducing the tolerance underflow bug ===\n")

ab = np.array([[1.5e-323, 4.9e-324],
               [9.9e-324, 0.0e+000]])

print(f"Matrix AB:\n{ab}\n")

u, s, vh = np.linalg.svd(ab)
print(f"Singular values: {s}")

default_tol = max(ab.shape) * np.max(s) * np.finfo(ab.dtype).eps
print(f"Default tolerance calculation: max(shape) * max(s) * eps")
print(f"  = {max(ab.shape)} * {np.max(s)} * {np.finfo(ab.dtype).eps}")
print(f"  = {default_tol}")
print(f"Underflowed to 0.0: {default_tol == 0.0}")

rank_default = np.linalg.matrix_rank(ab)
rank_explicit = np.linalg.matrix_rank(ab, tol=1e-15)
rank_explicit_tiny = np.linalg.matrix_rank(ab, tol=1e-300)

print(f"\nRank with default tol: {rank_default}")
print(f"Rank with tol=1e-15: {rank_explicit}")
print(f"Rank with tol=1e-300: {rank_explicit_tiny}")

# Additional analysis
print(f"\n=== Additional Analysis ===")
print(f"np.finfo(np.float64).eps: {np.finfo(np.float64).eps}")
print(f"np.finfo(np.float64).tiny: {np.finfo(np.float64).tiny}")
print(f"np.finfo(np.float64).smallest_subnormal: {np.finfo(np.float64).smallest_subnormal}")
print(f"\nSingular values compared to thresholds:")
for i, sv in enumerate(s):
    print(f"  s[{i}] = {sv:.3e}")
    print(f"    > 0.0? {sv > 0.0}")
    print(f"    > 1e-15? {sv > 1e-15}")
    print(f"    > 1e-300? {sv > 1e-300}")