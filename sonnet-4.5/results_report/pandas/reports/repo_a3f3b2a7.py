from pandas.io.formats.format import _trim_zeros_complex

# Test with complex numbers that have both real and imaginary parts
values = [complex(1, 2), complex(3, 4)]
str_values = [str(v) for v in values]

print(f"Input:  {str_values}")
result = _trim_zeros_complex(str_values)
print(f"Output: {result}")

# Let's also test with a single value to be more explicit
single_value = complex(1.0, 1.0)
single_str = str(single_value)
print(f"\nSingle Input:  '{single_str}'")
single_result = _trim_zeros_complex([single_str])
print(f"Single Output: '{single_result[0]}'")

# Check if parenthesis was lost
if single_str.endswith(')') and not single_result[0].endswith(')'):
    print("\nERROR: Closing parenthesis was removed!")
    print(f"  Expected: '{single_str}'")
    print(f"  Got:      '{single_result[0]}'")