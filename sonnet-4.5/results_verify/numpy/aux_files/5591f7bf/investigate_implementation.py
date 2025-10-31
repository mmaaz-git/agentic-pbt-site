import numpy as np
import numpy.polynomial as poly
import inspect

# Let's investigate the implementation details
p = poly.Polynomial([0.0, 1e-308])

print("Investigating Polynomial implementation:")
print("=" * 60)

# Check if there's a trim method and what it does
print("Has trim method:", hasattr(p, 'trim'))
if hasattr(p, 'trim'):
    print("Trim method doc:", p.trim.__doc__)
    print()

# Let's trace what happens during operations
print("Testing with very small coefficient 1e-308:")
print("-" * 60)

# Original polynomial
print(f"Original p.coef: {p.coef}")

# Multiplication
p_mult = p * p
print(f"After p * p: {p_mult.coef}")

# Power
p_power = p ** 2
print(f"After p ** 2: {p_power.coef}")

# Let's see what happens when we manually square the coefficients
print("\nManual calculation:")
coef_squared = 1e-308 * 1e-308
print(f"1e-308 * 1e-308 = {coef_squared}")
print(f"Is this underflow to zero? {coef_squared == 0.0}")

# Let's test with a slightly larger value
print("\nTesting with 1e-154 (sqrt of 1e-308):")
print("-" * 60)
p2 = poly.Polynomial([0.0, 1e-154])
print(f"Original p2.coef: {p2.coef}")

p2_mult = p2 * p2
print(f"After p2 * p2: {p2_mult.coef}")

p2_power = p2 ** 2
print(f"After p2 ** 2: {p2_power.coef}")
print(f"1e-154 * 1e-154 = {1e-154 * 1e-154}")

# Let's check what the trim method does
print("\nTesting trim method:")
print("-" * 60)
test_poly = poly.Polynomial([0.0, 0.0, 1.0, 0.0, 0.0])
print(f"Before trim: {test_poly.coef}")
trimmed = test_poly.trim()
print(f"After trim: {trimmed.coef}")

test_poly2 = poly.Polynomial([0.0, 0.0, 0.0])
print(f"All zeros before trim: {test_poly2.coef}")
trimmed2 = test_poly2.trim()
print(f"All zeros after trim: {trimmed2.coef}")

# Let's check how multiplication handles trimming
print("\nChecking multiplication trimming behavior:")
print("-" * 60)
p3 = poly.Polynomial([1.0, 0.0])
p4 = poly.Polynomial([1.0, 0.0])
mult_result = p3 * p4
print(f"[1, 0] * [1, 0] = {mult_result.coef}")

# And power
power_result = p3 ** 2
print(f"[1, 0] ** 2 = {power_result.coef}")

# Check if they're using different code paths
print("\nChecking method types:")
print("-" * 60)
print(f"Type of __mul__: {type(p.__mul__)}")
print(f"Type of __pow__: {type(p.__pow__)}")

# Let's look at the class hierarchy
print(f"\nPolynomial class MRO: {poly.Polynomial.__mro__}")

# Test mathematical equivalence
print("\nTesting mathematical equivalence:")
print("-" * 60)
p_test = poly.Polynomial([0.0, 1e-308])
p_mult = p_test * p_test
p_power = p_test ** 2

# Evaluate at a point to check mathematical equivalence
x = 1.0
val_mult = p_mult(x)
val_power = p_power(x)
print(f"p*p evaluated at x=1: {val_mult}")
print(f"p**2 evaluated at x=1: {val_power}")
print(f"Are they mathematically equal? {val_mult == val_power}")

# But the coefficient arrays are different
print(f"But coefficient arrays differ: {p_mult.coef} vs {p_power.coef}")