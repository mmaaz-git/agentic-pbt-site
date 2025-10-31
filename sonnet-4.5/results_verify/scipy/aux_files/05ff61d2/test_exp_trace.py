import numpy.f2py.symbolic as sym

# Let's trace the parsing step by step
# Start with 'x**2'
s = 'x**2'

# This gets transformed to 'x@__f2py_DOUBLE_STAR@2'
# Then when checking multiplication split, it stays as ['x@__f2py_DOUBLE_STAR@2']
# So it doesn't enter the multiplication branch

# Then checks '**' in r
# But r is 'x@__f2py_DOUBLE_STAR@2' so '**' is NOT in r!

test = 'x@__f2py_DOUBLE_STAR@2'
print(f"'**' in '{test}':", '**' in test)

# So it won't enter the exponentiation branch either!
# Let's trace what happens then...

# It might just parse 'x@__f2py_DOUBLE_STAR@2' as a symbol!
expr = sym.fromstring('x@__f2py_DOUBLE_STAR@2')
print(f"\nParsing 'x@__f2py_DOUBLE_STAR@2': {expr}")
print(f"Op: {expr.op}")
print(f"Data: {expr.data}")