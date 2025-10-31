import numpy as np
from scipy.optimize import newton

def f(x):
    return x**2 - 4

def fprime(x):
    return 2 * x

# Test with positive rtol
print("Testing with positive rtol=0.1:")
try:
    root = newton(f, 1.0, fprime=fprime, rtol=0.1, full_output=True, disp=False)
    print(f"  Result: {root}")
except Exception as e:
    print(f"  Error: {e}")

# Test with negative rtol
print("\nTesting with negative rtol=-0.1:")
try:
    root = newton(f, 1.0, fprime=fprime, rtol=-0.1, full_output=True, disp=False)
    print(f"  Result: {root}")
except Exception as e:
    print(f"  Error: {e}")

# Test what np.isclose does with negative rtol
print("\nTesting np.isclose with negative rtol:")
a = 2.0
b = 2.1
print(f"  np.isclose({a}, {b}, rtol=0.1): {np.isclose(a, b, rtol=0.1)}")
print(f"  np.isclose({a}, {b}, rtol=-0.1): {np.isclose(a, b, rtol=-0.1)}")

# Check the formula: |a - b| <= atol + rtol * max(|a|, |b|)
print("\nManual check of convergence formula with negative rtol:")
atol = 1.48e-8  # default tol in newton
rtol_neg = -0.1
diff = abs(a - b)
threshold = atol + rtol_neg * max(abs(a), abs(b))
print(f"  |{a} - {b}| = {diff}")
print(f"  Threshold = {atol} + {rtol_neg} * {max(abs(a), abs(b))} = {threshold}")
print(f"  Would converge? {diff <= threshold}")