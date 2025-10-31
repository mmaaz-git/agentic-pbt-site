import numpy as np
from numpy.polynomial import Polynomial

a = Polynomial([0., 1.])
b = Polynomial([1.0, 2.22507386e-311])

q, r = divmod(a, b)

print("Quotient:", q.coef)
print("Remainder:", r.coef)
print("Contains inf:", np.any(np.isinf(q.coef)) or np.any(np.isinf(r.coef)))