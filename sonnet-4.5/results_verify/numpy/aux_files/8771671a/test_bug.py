import numpy.f2py.symbolic as sym

# Test the main failing case
print("Testing '(' input:")
try:
    expr = sym.fromstring("(")
    print(f"Parsed: {expr}")
except RecursionError as e:
    print(f"RecursionError raised (BUG)")
except ValueError as e:
    print(f"ValueError raised (expected): {e}")

# Test other unbalanced inputs mentioned
test_cases = ["(", "((", "(()", ")"]
print("\nTesting various unbalanced inputs:")
for test in test_cases:
    print(f"\nInput: '{test}'")
    try:
        expr = sym.fromstring(test)
        print(f"  Parsed successfully: {expr}")
    except RecursionError:
        print(f"  RecursionError (BUG)")
    except ValueError as e:
        print(f"  ValueError (expected): {e}")
    except Exception as e:
        print(f"  Other exception: {type(e).__name__}: {e}")