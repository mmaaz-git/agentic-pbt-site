from numpy.polynomial import Polynomial
import numpy as np

# Use the failing case found by Hypothesis
coeffs = [0.0, 2.2250738585072014e-308]
n = 2

print(f"Testing polynomial with coefficients: {coeffs}")
print(f"Raising to power: {n}")
print()

# Create the polynomial
p = Polynomial(coeffs)

# Compute p**2 using power operator
power_result = p ** n

# Compute p*p using multiplication
mult_result = Polynomial([1])
for _ in range(n):
    mult_result = mult_result * p

print("Original polynomial p:")
print(f"  Coefficients: {p.coef}")
print(f"  Shape: {p.coef.shape}")
print()

print(f"Result of p**{n}:")
print(f"  Coefficients: {power_result.coef}")
print(f"  Shape: {power_result.coef.shape}")
print()

print(f"Result of p*p (repeated multiplication):")
print(f"  Coefficients: {mult_result.coef}")
print(f"  Shape: {mult_result.coef.shape}")
print()

print("Comparison:")
print(f"  Shapes match? {power_result.coef.shape == mult_result.coef.shape}")
print(f"  p**{n} == p*p? {power_result == mult_result}")

# Show that the polynomials evaluate to the same values even though they're not "equal"
test_values = [0, 1, -1, 2, -2]
print("\nEvaluation at test points:")
for x in test_values:
    print(f"  At x={x:3}: p**{n}(x) = {power_result(x):.2e}, p*p(x) = {mult_result(x):.2e}")