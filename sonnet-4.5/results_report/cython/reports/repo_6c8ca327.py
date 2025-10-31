from Cython.Utils import normalise_float_repr

# Test case from the bug report
float_str = '-7.941487302529372e-299'
result = normalise_float_repr(float_str)

print(f"Input:  {float_str}")
print(f"Result: {result!r}")

# Try to parse it back
try:
    result_value = float(result)
    print(f"Parsed value: {result_value}")
except ValueError as e:
    print(f"Error parsing result: {e}")