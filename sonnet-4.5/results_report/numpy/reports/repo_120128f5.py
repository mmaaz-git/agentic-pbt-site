import numpy.f2py.symbolic as symbolic

# Test case that crashes with unmatched quote
print("Testing eliminate_quotes with single unmatched double quote:")
print("Input: '\"'")
try:
    result = symbolic.eliminate_quotes('"')
    print(f"Result: {result}")
except AssertionError as e:
    print("AssertionError raised")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting eliminate_quotes with single unmatched single quote:")
print("Input: \"'\"")
try:
    result = symbolic.eliminate_quotes("'")
    print(f"Result: {result}")
except AssertionError as e:
    print("AssertionError raised")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")