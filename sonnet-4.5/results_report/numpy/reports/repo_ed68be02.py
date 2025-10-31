import numpy as np
import numpy.char as char

# Test the specific Unicode character mentioned in the bug report
arr = np.array(['ŉabc'], dtype=str)
result = char.capitalize(arr)

print(f"Input: {arr[0]!r}")
print(f"NumPy result: {result[0]!r}")
print(f"Python's str.capitalize result: {'ŉabc'.capitalize()!r}")
print(f"Input length: {len(arr[0])}")
print(f"NumPy result length: {len(result[0])}")
print(f"Expected length: {len('ŉabc'.capitalize())}")

# Verify the assertion fails
try:
    assert result[0] == 'ʼNabc'
    print("Assertion passed (unexpected)")
except AssertionError:
    print(f"Assertion failed: {result[0]!r} != 'ʼNabc'")