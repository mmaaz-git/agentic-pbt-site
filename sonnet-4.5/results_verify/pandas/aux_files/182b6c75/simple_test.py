from pandas.api import types as pat

print("Testing is_re_compilable with ')':")
try:
    result = pat.is_re_compilable(')')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting is_re_compilable with '(':")
try:
    result = pat.is_re_compilable('(')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting is_re_compilable with '?':")
try:
    result = pat.is_re_compilable('?')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting is_re_compilable with '.*' (valid regex):")
try:
    result = pat.is_re_compilable('.*')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting is_re_compilable with 1 (non-string):")
try:
    result = pat.is_re_compilable(1)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")