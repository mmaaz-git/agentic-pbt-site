import pandas.api.types as pt

print("Testing pt.is_re_compilable('['):")
try:
    result = pt.is_re_compilable("[")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting pt.is_re_compilable(')'):")
try:
    result = pt.is_re_compilable(")")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting pt.is_re_compilable('?'):")
try:
    result = pt.is_re_compilable("?")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting pt.is_re_compilable('\\\\'):")
try:
    result = pt.is_re_compilable("\\")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting valid regex pt.is_re_compilable('.*'):")
try:
    result = pt.is_re_compilable(".*")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting non-string pt.is_re_compilable(123):")
try:
    result = pt.is_re_compilable(123)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")