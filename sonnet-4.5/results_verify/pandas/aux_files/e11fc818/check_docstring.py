import pandas.api.types as pt

# Get the docstring
print("pandas_dtype docstring:")
print("=" * 60)
print(pt.pandas_dtype.__doc__)
print("=" * 60)

# Also check if there's help available
import inspect
print("\nFunction signature:")
print(inspect.signature(pt.pandas_dtype))