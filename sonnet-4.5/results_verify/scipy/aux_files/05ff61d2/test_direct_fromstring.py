import numpy.f2py.symbolic as sym
from numpy.f2py.symbolic import Language

# Test directly
e1 = sym.fromstring('x**2', language=Language.Fortran)
print(f"Fortran: x**2 -> {e1} -> tostring: {e1.tostring()}")

e2 = sym.fromstring('x**2', language=Language.Python)
print(f"Python: x**2 -> {e2} -> tostring: {e2.tostring()}")

e3 = sym.fromstring('x**2', language=Language.C)
print(f"C: x**2 -> {e3} -> tostring: {e3.tostring()}")

# The default language
e4 = sym.fromstring('x**2')
print(f"Default (C): x**2 -> {e4} -> tostring: {e4.tostring()}")