import numpy as np
from numpy.polynomial import Polynomial

# Check what divmod actually calls
print("divmod method:", Polynomial.__divmod__)

# Check the implementation details
import inspect
print("\nSource of divmod:")
try:
    print(inspect.getsource(Polynomial.__divmod__))
except:
    print("Could not get source")

# Let's check if divmod for Polynomial calls polydiv
a = Polynomial([0., 1.])
b = Polynomial([1.0, 1.0])  # Use a normal value first

q, r = divmod(a, b)
print("\nNormal division works:")
print("Quotient:", q.coef)
print("Remainder:", r.coef)