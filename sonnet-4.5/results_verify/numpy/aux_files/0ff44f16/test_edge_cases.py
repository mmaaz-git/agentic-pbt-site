from numpy.f2py import symbolic

test_cases = [
    '(',     # Single opening parenthesis
    '[',     # Single opening bracket
    '{',     # Single opening brace
    '(/',    # Opening rounddiv
    ')',     # Single closing parenthesis
    '()',    # Balanced parentheses
    '(())',  # Nested balanced
    '(()',   # Unbalanced nested
    '())',   # Extra closing
    'a(b',   # Text with unbalanced
    '(a+b',  # Expression with unbalanced
]

for test_str in test_cases:
    print(f"\nTesting: {test_str!r}")
    try:
        result, mapping = symbolic.replace_parenthesis(test_str)
        print(f"  Success: new_s={result!r}, mapping={mapping}")
        # Try to reconstruct
        reconstructed = symbolic.unreplace_parenthesis(result, mapping)
        if reconstructed == test_str:
            print(f"  Round-trip successful!")
        else:
            print(f"  Round-trip failed: {reconstructed!r} != {test_str!r}")
    except RecursionError:
        print(f"  RecursionError!")
    except ValueError as e:
        print(f"  ValueError: {e}")
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")