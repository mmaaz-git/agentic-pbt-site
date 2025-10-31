import math
from starlette.convertors import FloatConvertor

convertor = FloatConvertor()
x = 1e-21

string_repr = convertor.to_string(x)
result = convertor.convert(string_repr)

print(f"Original: {x}")
print(f"String: '{string_repr}'")
print(f"Result: {result}")
print(f"Match: {result == x}")

# Also test the specific failing example from the bug report
x2 = 5.08183882917904e-140
string_repr2 = convertor.to_string(x2)
result2 = convertor.convert(string_repr2)

print(f"\nOriginal: {x2}")
print(f"String: '{string_repr2}'")
print(f"Result: {result2}")
print(f"Match: {result2 == x2}")