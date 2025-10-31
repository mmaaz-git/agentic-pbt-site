from Cython.Utils import normalise_float_repr

x = "6.103515625e-05"
result = normalise_float_repr(x)
print(f"Input:  {x}")
print(f"Output: {result}")
print(f"Input value:  {float(x)}")
print(f"Output value: {float(result)}")
print(f"Expected value: {6.103515625e-05}")
print(f"Error factor: {float(result) / float(x)}")