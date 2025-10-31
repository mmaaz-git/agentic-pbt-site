import numpy as np
import numpy.char as char

# Test the specific example from the bug report
arr = np.array(['ŉabc'], dtype=str)
result = char.capitalize(arr)

print(f"Input string: {'ŉabc'!r}")
print(f"Input array dtype: {arr.dtype}")
print(f"Result from numpy: {result[0]!r}")
print(f"Expected (Python str.capitalize): {'ŉabc'.capitalize()!r}")
print(f"Length of result: {len(result[0])}")
print(f"Length of expected: {len('ŉabc'.capitalize())}")

# Check if they match
if result[0] == 'ŉabc'.capitalize():
    print("✓ Results match")
else:
    print("✗ Results DO NOT match - truncation occurred!")

# Also test the character conversion mentioned
print(f"\nCharacter 'ŉ' info:")
print(f"  Unicode: U+{ord('ŉ'):04X}")
print(f"  Capitalized: {'ŉ'.capitalize()!r}")
print(f"  Length after capitalize: {len('ŉ'.capitalize())}")