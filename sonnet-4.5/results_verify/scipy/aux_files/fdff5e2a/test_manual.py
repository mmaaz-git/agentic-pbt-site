import scipy.optimize.cython_optimize._zeros as zeros

# Test 1: Wrong order (descending - what users might naturally try)
print("Test 1: Using descending order coefficients (1, 0, 0, -2) for x^3 - 2:")
args = (1.0, 0.0, 0.0, -2.0)
result = zeros.EXAMPLES_MAP['brentq'](args, 1.0, 2.0, 1e-6, 1e-6, 100)

print(f"Expected root of x^3 - 2: {2**(1/3)}")
print(f"Got: {result}")

# Verify this is wrong
poly_val_wrong = 1.0 * result**3 + 0.0 * result**2 + 0.0 * result - 2.0
print(f"Polynomial value at 'root' (using x^3 - 2): {poly_val_wrong}")

# What the function actually computed
actual_poly = 1.0 + 0.0*result + 0.0*result**2 - 2.0*result**3
print(f"What function actually computed (1 + 0*x + 0*x^2 - 2*x^3): {actual_poly}")

print("\n" + "="*50 + "\n")

# Test 2: Correct order (ascending)
print("Test 2: Using ascending order coefficients (-2, 0, 0, 1) for c0 + c1*x + c2*x^2 + c3*x^3:")
args = (-2.0, 0.0, 0.0, 1.0)
result2 = zeros.EXAMPLES_MAP['brentq'](args, 1.0, 2.0, 1e-6, 1e-6, 100)

print(f"Expected root of x^3 - 2: {2**(1/3)}")
print(f"Got: {result2}")

# Verify this is correct
poly_val_correct = -2.0 + 0.0*result2 + 0.0*result2**2 + 1.0*result2**3
print(f"Polynomial value at root (-2 + 0*x + 0*x^2 + 1*x^3): {poly_val_correct}")