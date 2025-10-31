import numpy as np
import numpy.polynomial as poly

print("Simple reproduction of the bug:")
print("=" * 50)

p = poly.Polynomial([0.0, 1e-308])

result_power = p ** 2
result_mult = p * p

print(f'Original polynomial coefficients: {p.coef}')
print(f'p**2 coefficients: {result_power.coef}')
print(f'p*p coefficients: {result_mult.coef}')
print(f'Arrays equal: {np.array_equal(result_power.coef, result_mult.coef)}')
print()

# Let's also test with Chebyshev polynomials as mentioned in the report
print("Testing with Chebyshev polynomial:")
print("-" * 50)
from numpy.polynomial import Chebyshev

c = Chebyshev([0.0, 1e-308])
c_power = c ** 2
c_mult = c * c

print(f'Original Chebyshev coefficients: {c.coef}')
print(f'c**2 coefficients: {c_power.coef}')
print(f'c*c coefficients: {c_mult.coef}')
print(f'Arrays equal: {np.array_equal(c_power.coef, c_mult.coef)}')
print()

# Let's test what happens with other polynomial types mentioned
print("Testing with other polynomial types:")
print("-" * 50)

from numpy.polynomial import Legendre, Hermite, HermiteE, Laguerre

poly_types = [
    ("Legendre", Legendre),
    ("Hermite", Hermite),
    ("HermiteE", HermiteE),
    ("Laguerre", Laguerre)
]

for name, PolyClass in poly_types:
    p = PolyClass([0.0, 1e-308])
    p_power = p ** 2
    p_mult = p * p
    equal = np.array_equal(p_power.coef, p_mult.coef)
    print(f'{name}: p**2 shape={p_power.coef.shape}, p*p shape={p_mult.coef.shape}, equal={equal}')