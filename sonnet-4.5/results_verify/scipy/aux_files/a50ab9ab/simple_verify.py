#!/usr/bin/env python3
"""Simple verification of the mathematical issue."""

import scipy.special as sp
import math

print("MATHEMATICAL VERIFICATION")
print("=" * 60)

x = 2.0
lmbda = 5e-324  # Subnormal value

# Forward transformation
y = sp.boxcox(x, lmbda)
print(f"boxcox({x}, {lmbda}) = {y}")
print(f"log({x}) = {math.log(x)}")
print(f"boxcox returns log(x): {y == math.log(x)}")
print()

# Inverse transformation
x_recovered = sp.inv_boxcox(y, lmbda)
print(f"inv_boxcox({y}, {lmbda}) = {x_recovered}")
print(f"exp({y}) = {math.exp(y)}")
print(f"Should return {x} but returns {x_recovered}")
print()

# The issue
print("THE ISSUE:")
print(f"- boxcox treats λ={lmbda} as 0 and returns log(x)")
print(f"- inv_boxcox does NOT treat λ={lmbda} as 0")
print(f"- This breaks the round-trip property")
print(f"- Error: {abs(x - x_recovered)}")