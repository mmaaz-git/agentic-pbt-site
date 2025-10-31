import numpy.f2py.symbolic as sym
from numpy.f2py.symbolic import Language

# Demonstrate the division round-trip bug
e1 = sym.fromstring('x / y')
print(f'Original expression: {e1}')
print(f'Type of e1: {type(e1)}')
print(f'e1.op: {e1.op}')
print(f'e1.data: {e1.data}')
print()

# Show tostring output for different languages
print('tostring() outputs by language:')
print(f'  Fortran: {e1.tostring(language=Language.Fortran)}')
print(f'  Python:  {e1.tostring(language=Language.Python)}')
print(f'  C:       {e1.tostring(language=Language.C)}')
print()

# Try to parse back the Fortran output
fortran_string = e1.tostring(language=Language.Fortran)
print(f'Attempting to re-parse Fortran output: "{fortran_string}"')
try:
    e2 = sym.fromstring(fortran_string)
    print(f'Re-parsed expression: {e2}')
    print(f'Type of e2: {type(e2)}')
    print(f'e2.op: {e2.op}')
    print(f'e2.data: {e2.data}')
    print(f'Are they equal? {e1 == e2}')
    if e1 != e2:
        print(f'FAILURE: Round-trip failed!')
        print(f'  e1.data[0]: {e1.data[0]}, type: {type(e1.data[0])}')
        print(f'  e2.data[0]: {e2.data[0]}, type: {type(e2.data[0])}')
except Exception as ex:
    print(f'ERROR during re-parsing: {ex}')
print()

# Try the same with Python language mode
python_string = e1.tostring(language=Language.Python)
print(f'Attempting to re-parse Python output: "{python_string}"')
try:
    e3 = sym.fromstring(python_string, language=Language.Python)
    print(f'Re-parsed expression: {e3}')
    print(f'Are they equal? {e1 == e3}')
    if e1 != e3:
        print(f'FAILURE: Round-trip failed for Python mode!')
except Exception as ex:
    print(f'ERROR during re-parsing: {ex}')
print()

# Try with C mode
c_string = e1.tostring(language=Language.C)
print(f'Attempting to re-parse C output: "{c_string}"')
try:
    e4 = sym.fromstring(c_string, language=Language.C)
    print(f'Re-parsed expression: {e4}')
    print(f'Are they equal? {e1 == e4}')
    if e1 == e4:
        print(f'SUCCESS: Round-trip works in C mode!')
except Exception as ex:
    print(f'ERROR during re-parsing: {ex}')