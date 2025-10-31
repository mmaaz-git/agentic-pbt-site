import pandas.api.types as pat

print("Testing is_re_compilable('[') ...")
try:
    result = pat.is_re_compilable('[')
    print(f"Result: {result}")
except Exception as e:
    print(f"Raised {type(e).__name__}: {e}")

print("\nTesting is_re_compilable('?') ...")
try:
    result = pat.is_re_compilable('?')
    print(f"Result: {result}")
except Exception as e:
    print(f"Raised {type(e).__name__}: {e}")

print("\nTesting is_re_compilable('(unclosed') ...")
try:
    result = pat.is_re_compilable('(unclosed')
    print(f"Result: {result}")
except Exception as e:
    print(f"Raised {type(e).__name__}: {e}")

# Test with valid regex
print("\nTesting is_re_compilable('abc') (valid regex) ...")
try:
    result = pat.is_re_compilable('abc')
    print(f"Result: {result}")
except Exception as e:
    print(f"Raised {type(e).__name__}: {e}")

# Test with non-string type
print("\nTesting is_re_compilable(123) (integer) ...")
try:
    result = pat.is_re_compilable(123)
    print(f"Result: {result}")
except Exception as e:
    print(f"Raised {type(e).__name__}: {e}")