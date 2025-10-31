import numpy as np
import numpy.char as char

arr = np.array(['ß'], dtype=str)
result = char.upper(arr)

print(f"Result: {result[0]!r}")
print(f"Expected: {'ß'.upper()!r}")

# Check the dtype
print(f"\nInput array dtype: {arr.dtype}")
print(f"Output array dtype: {result.dtype}")

# Test more cases that expand
test_cases = ['ß', 'ﬁ', 'ﬂ', 'ﬃ', 'ﬄ', 'ﬅ', 'ﬆ']
for test_char in test_cases:
    arr = np.array([test_char], dtype=str)
    result = char.upper(arr)
    expected = test_char.upper()
    print(f"\nInput: {test_char!r} -> numpy result: {result[0]!r}, expected: {expected!r}, match: {result[0] == expected}")

# The assertion from the bug report
arr = np.array(['ß'], dtype=str)
result = char.upper(arr)
try:
    assert result[0] == 'SS'
    print("\n✓ Assertion passed")
except AssertionError:
    print(f"\n✗ Assertion failed: {result[0]!r} != 'SS'")