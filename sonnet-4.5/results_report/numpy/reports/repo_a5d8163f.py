#!/usr/bin/env python3
"""Demonstrate the numpy.f2py.symbolic power operator bug"""

import numpy.f2py.symbolic as sym
from numpy.f2py.symbolic import Language

# Test case from bug report
e = sym.fromstring('x**2')
print(f'Input: x**2')
print(f'Parsed expression: {e!r}')
print(f'Fortran output: {e.tostring(language=Language.Fortran)}')
print(f'Python output: {e.tostring(language=Language.Python)}')
print(f'C output: {e.tostring(language=Language.C)}')
print()

# Test with explicit language specification
print('With explicit Fortran language:')
e_fortran = sym.fromstring('x**2', language=Language.Fortran)
print(f'Input: x**2 (parsed as Fortran)')
print(f'Parsed expression: {e_fortran!r}')
print(f'Fortran output: {e_fortran.tostring(language=Language.Fortran)}')
print()

# Test round-trip
print('Round-trip test:')
malformed = e.tostring(language=Language.Fortran)
print(f'Malformed string: {malformed!r}')
e2 = sym.fromstring(malformed)
print(f'Re-parsed: {e2!r}')
print(f'Re-parsed tostring: {e2.tostring(language=Language.Fortran)}')
print(f'Round-trip equality: {e == e2}')