import pandas.api.types as types
import inspect

# Get the docstring for infer_dtype
print("=== DOCSTRING FOR infer_dtype ===\n")
print(types.infer_dtype.__doc__)
print("\n=== SIGNATURE ===\n")
print(inspect.signature(types.infer_dtype))