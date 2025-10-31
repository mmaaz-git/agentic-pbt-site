import numpy as np
import numpy.polynomial as np_poly

# Test with the specific failing input from the bug report
print("Testing with specific failing input from bug report...")
coef = [0.0, 1.0, 3.254353641323301e-273]
p = np_poly.Polynomial(coef)

print(f"Polynomial coefficients: {coef}")
print(f"Polynomial degree: {p.degree()}")

roots = p.roots()
print(f"Computed roots: {roots}")

for root in roots:
    value = p(root)
    print(f"p({root}) = {value}")
    abs_value = abs(value)
    if abs_value >= 1e-6:
        print(f"ERROR: Root evaluation is {abs_value}, which is >= 1e-6")

print("\n" + "="*50 + "\n")

# Test with the reproducing example from the bug report
print("Testing with the reproducing example from the bug report...")
p2 = np_poly.Polynomial([1.0, 1.0, 3.9968426114653685e-66])

roots2 = p2.roots()
print(f'Computed roots: {roots2}')

for root in roots2:
    print(f'p({root}) = {p2(root)}')

# Check if p(-1.0) = 0 as claimed
print(f"\nVerifying p(-1.0) = {p2(-1.0)}")