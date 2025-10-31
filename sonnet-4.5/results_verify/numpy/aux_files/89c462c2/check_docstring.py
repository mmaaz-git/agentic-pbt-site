import numpy.rec
import inspect

# Get the docstring for field method
rec_arr = numpy.rec.fromarrays([[1, 2]], names='x')
print("=== recarray.field docstring ===")
print(rec_arr.field.__doc__)

print("\n=== Check method signature ===")
sig = inspect.signature(rec_arr.field)
print(f"Signature: {sig}")

print("\n=== Test behavior ===")
# Test with valid index
print(f"field(0) returns: {rec_arr.field(0)}")

# Test with field name
print(f"field('x') returns: {rec_arr.field('x')}")

# Check numpy version
import numpy as np
print(f"\nNumPy version: {np.__version__}")