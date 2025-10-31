import numpy.f2py.symbolic as symbolic

# Test with default language (C)
expr = symbolic.as_symbol('a') * symbolic.as_symbol('a')
print(f'Original: {repr(expr)}')
s = expr.tostring()  # default is Fortran
print(f'tostring() with default language: "{s}"')

# Parse with default language (C)
parsed = symbolic.Expr.parse(s)  # default is C
print(f'Parsed with default language (C): {repr(parsed)}')
print(f'Round-trip successful: {parsed == expr}')

# Parse with Fortran language
parsed_fortran = symbolic.Expr.parse(s, language=symbolic.Language.Fortran)
print(f'\nParsed with Fortran language: {repr(parsed_fortran)}')
print(f'Round-trip successful with Fortran: {parsed_fortran == expr}')