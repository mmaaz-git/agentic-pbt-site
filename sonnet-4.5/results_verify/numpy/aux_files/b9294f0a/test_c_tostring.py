import numpy.f2py.symbolic as symbolic

# Test with C language output
expr = symbolic.as_symbol('a') * symbolic.as_symbol('a')
print(f'Original: {repr(expr)}')

# tostring with Fortran (default)
s_fortran = expr.tostring(language=symbolic.Language.Fortran)
print(f'tostring() Fortran: "{s_fortran}"')

# tostring with C
s_c = expr.tostring(language=symbolic.Language.C)
print(f'tostring() C: "{s_c}"')

# Parse C string with C language
parsed_c = symbolic.Expr.parse(s_c, language=symbolic.Language.C)
print(f'\nParsed C format with C parser: {repr(parsed_c)}')
print(f'Round-trip successful: {parsed_c == expr}')

# Parse Fortran string with Fortran language
parsed_fortran = symbolic.Expr.parse(s_fortran, language=symbolic.Language.Fortran)
print(f'\nParsed Fortran format with Fortran parser: {repr(parsed_fortran)}')
print(f'Round-trip successful: {parsed_fortran == expr}')