from pandas.core.computation.eval import _check_expression

# Test cases
test_cases = [
    ("", 'empty string'),
    (" ", 'single space'),
    ("\t", 'tab character'),
    ("\n", 'newline character'),
    ("\r", 'carriage return'),
    ("  \t\n  ", 'mixed whitespace')
]

for expr, description in test_cases:
    print(f"Testing {description} (repr: {repr(expr)}):")
    try:
        _check_expression(expr)
        print(f"  Passed check (no exception raised)")
    except ValueError as e:
        print(f"  ValueError: {e}")
    print()