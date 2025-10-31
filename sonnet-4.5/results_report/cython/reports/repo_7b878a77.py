from Cython.Utils import normalise_float_repr

f1 = 5.960464477539063e-08
str1 = str(f1)
result1 = normalise_float_repr(str1)
print(f"Input: {f1}")
print(f"Input string: {str1}")
print(f"Result: {result1}")
try:
    parsed = float(result1)
    print(f"Float of result: {parsed}")
    if parsed != f1:
        print(f"BUG: Expected {f1}, got {parsed}")
except ValueError as e:
    print(f"BUG: Cannot parse result as float: {e}")

print()

f2 = -3.0929648190816446e-178
str2 = str(f2)
result2 = normalise_float_repr(str2)
print(f"Input: {f2}")
print(f"Input string: {str2}")
print(f"Result: {result2}")
try:
    parsed = float(result2)
    print(f"Float of result: {parsed}")
    if parsed != f2:
        print(f"BUG: Expected {f2}, got {parsed}")
except ValueError as e:
    print(f"BUG: Cannot parse result as float: {e}")