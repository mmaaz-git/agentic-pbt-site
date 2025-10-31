import pandas.api.types as pt
import inspect

# Get the source file location
print("Source file location:")
print(inspect.getfile(pt.pandas_dtype))

# Try to get the source code
print("\nSource code:")
print("=" * 60)
try:
    print(inspect.getsource(pt.pandas_dtype))
except:
    print("Could not get source code directly")