import pandas.io.formats.format as fmt

formatter = fmt.EngFormatter(accuracy=1, use_eng_prefix=False)

# Test with a very small number that's outside the [-24, 24] exponent range
num = 1e-50
formatted = formatter(num)
parsed = float(formatted)

print(f"Original: {num}")
print(f"Formatted: '{formatted}'")
print(f"Parsed back: {parsed}")
print(f"Data loss: {parsed == 0.0 and num != 0.0}")

# Test with the specific failing input from the bug report
num2 = 2.844615157173927e-200
formatted2 = formatter(num2)
parsed2 = float(formatted2)

print(f"\nOriginal: {num2}")
print(f"Formatted: '{formatted2}'")
print(f"Parsed back: {parsed2}")
print(f"Data loss: {parsed2 == 0.0 and num2 != 0.0}")

# Test with a very large number outside the range
num3 = 1e50
formatted3 = formatter(num3)
parsed3 = float(formatted3)

print(f"\nOriginal: {num3}")
print(f"Formatted: '{formatted3}'")
print(f"Parsed back: {parsed3}")
print(f"Expected: 1e50, Got: {parsed3}")
print(f"Data corruption: {parsed3 != num3}")