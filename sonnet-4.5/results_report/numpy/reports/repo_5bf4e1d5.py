from numpy.f2py import symbolic

# Test with single double-quote
s = '"'
print(f"Testing with single double-quote: {s!r}")
try:
    new_s, mapping = symbolic.eliminate_quotes(s)
    print(f"Result: new_s={new_s!r}, mapping={mapping}")
except AssertionError as e:
    print(f"AssertionError raised")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test with single single-quote
s = "'"
print(f"Testing with single single-quote: {s!r}")
try:
    new_s, mapping = symbolic.eliminate_quotes(s)
    print(f"Result: new_s={new_s!r}, mapping={mapping}")
except AssertionError as e:
    print(f"AssertionError raised")
    import traceback
    traceback.print_exc()