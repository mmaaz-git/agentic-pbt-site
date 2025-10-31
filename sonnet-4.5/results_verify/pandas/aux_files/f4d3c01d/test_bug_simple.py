from pandas.io.formats.format import _trim_zeros_complex

values = [complex(1, 2), complex(3, 4)]
str_values = [str(v) for v in values]

print(f"Input:  {str_values}")
result = _trim_zeros_complex(str_values)
print(f"Output: {result}")

# Check if the closing parenthesis is lost
for original, trimmed in zip(str_values, result):
    if original.endswith(')') and not trimmed.endswith(')'):
        print(f"ERROR: Lost closing parenthesis: {original} -> {trimmed}")