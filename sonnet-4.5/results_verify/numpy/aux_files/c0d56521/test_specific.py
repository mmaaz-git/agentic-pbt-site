import numpy as np
from numpy.polynomial import Polynomial

p1 = Polynomial([1.0, 1.0])
p2 = Polynomial([1.0, 1.401298464324817e-45])

print("Dividing (1 + x) by (1 + 1.4e-45*x):")
print(f"p1 = {p1.coef}")
print(f"p2 = {p2.coef}")

q, r = divmod(p1, p2)
reconstructed = q * p2 + r

print("\nDivmod result:")
print(f"quotient  = {q.coef}")
print(f"remainder = {r.coef}")

print("\nChecking p1 = q*p2 + r:")
print(f"Expected: {p1.coef}")
print(f"Got:      {reconstructed.coef}")
print(f"BUG: Constant term is {reconstructed.coef[0]} instead of {p1.coef[0]}")