import numpy.f2py.symbolic as symbolic

expr = symbolic.as_symbol('a') * symbolic.as_symbol('a')
print(f'Original: {repr(expr)}')

s = expr.tostring()
print(f'tostring(): "{s}"')

parsed = symbolic.Expr.parse(s)
print(f'Parsed: {repr(parsed)}')

print(f'Round-trip successful: {parsed == expr}')

# Let's also check the exact details to understand better
print(f'\nExpected factors: {expr.data}')
print(f'Parsed factors: {parsed.data}')