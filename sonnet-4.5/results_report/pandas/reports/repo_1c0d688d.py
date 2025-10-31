from pandas.core import ops

# Test case that causes RecursionError
result = ops.kleene_and(False, True, None, None)
print(f"Result: {result}")