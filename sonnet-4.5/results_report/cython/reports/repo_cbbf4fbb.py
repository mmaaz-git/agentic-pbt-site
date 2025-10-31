from Cython.Utils import normalise_float_repr

x = "-3.833509682449162e-128"
result = normalise_float_repr(x)
print(f"Input:  {x}")
print(f"Output: {result}")
print()

# Try to convert back to float
try:
    converted = float(result)
    print(f"Successfully converted back to float: {converted}")
except ValueError as e:
    print(f"ValueError: {e}")