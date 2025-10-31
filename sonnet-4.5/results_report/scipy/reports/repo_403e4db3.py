from scipy.optimize import ridder

def f(x):
    return x * x - 2.0

print("Test case: Finding root of f(x) = x² - 2 in [-2, 1]")
print("Expected root: -√2 ≈ -1.41421356...")

result = ridder(f, -2.0, 1.0, xtol=1e-3, rtol=1e-3, full_output=True, disp=False)
root, info = result

print(f"Converged: {info.converged}")
print(f"Iterations: {info.iterations}")
print(f"Root: {root:.15f}")
print(f"f(root): {f(root):.2e}")

print("\nComparison: Same function, interval [0, 2] (positive root)")
result2 = ridder(f, 0.0, 2.0, xtol=1e-3, rtol=1e-3, full_output=True)
root2, info2 = result2
print(f"Converged: {info2.converged}")
print(f"Iterations: {info2.iterations}")
print(f"Root: {root2:.15f}")
print(f"f(root): {f(root2):.2e}")