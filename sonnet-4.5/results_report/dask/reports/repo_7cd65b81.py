#!/usr/bin/env python3

import numpy.f2py.symbolic as symbolic

# Create an expression that is a * a (which should be equivalent to a^2)
expr = symbolic.as_symbol('a') * symbolic.as_symbol('a')
print(f'Original expression: {repr(expr)}')

# Convert to string - this should produce "a ** 2"
s = expr.tostring()
print(f'tostring() output: "{s}"')

# Parse the string back
parsed = symbolic.Expr.parse(s)
print(f'Parsed expression: {repr(parsed)}')

# Check if round-trip is successful
print(f'Round-trip successful: {parsed == expr}')