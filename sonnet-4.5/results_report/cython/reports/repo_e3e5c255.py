from Cython.Utils import normalise_float_repr

print("Bug 1: Invalid float string for very small negative number")
print("-" * 60)
f1 = -1.670758163823954e-133
float_str1 = str(f1)
result1 = normalise_float_repr(float_str1)
print(f"Input:  {float_str1}")
print(f"Output: {result1}")
print(f"Attempting to convert back to float...")
try:
    converted = float(result1)
    print(f"Successfully converted: {converted}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\n" + "=" * 60 + "\n")

print("Bug 2: Value corruption for small numbers")
print("-" * 60)
f2 = 1.114036198514633e-05
float_str2 = str(f2)
result2 = normalise_float_repr(float_str2)
print(f"Input:  {float_str2} = {f2}")
print(f"Output: {result2} = {float(result2)}")
print(f"Error: {abs(float(result2) - f2) / abs(f2) * 100:.1f}%")

print("\n" + "=" * 60 + "\n")

print("Additional test cases")
print("-" * 60)
test_cases = [
    -1e-10,
    -1.0,
    -0.5,
    -10.5,
    -10000000000.0,
    0.0,
    1.0,
]

for f in test_cases:
    float_str = str(f)
    result = normalise_float_repr(float_str)
    print(f"Input: {float_str:20s} → Output: {result:20s}", end="")
    try:
        converted = float(result)
        if abs(converted - f) < 1e-10 or (f != 0 and abs((converted - f) / f) < 1e-10):
            print(" ✓ Correct")
        else:
            print(f" ✗ Wrong value: {converted}")
    except ValueError:
        print(" ✗ Invalid float")