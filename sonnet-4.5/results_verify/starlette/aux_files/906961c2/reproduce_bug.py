import numpy as np
import numpy.char as char

# Test the specific failing input from the bug report
arr = np.array(['ﬁ test'], dtype=str)
result = char.title(arr)

print(f"Input: {arr[0]!r}")
print(f"Result: {result[0]!r}")
print(f"Expected: {'ﬁ test'.title()!r}")
print(f"Input length: {len(arr[0])}")
print(f"Result length: {len(result[0])}")
print(f"Expected length: {len('ﬁ test'.title())}")
print(f"Input dtype: {arr.dtype}")
print(f"Result dtype: {result.dtype}")

# Check if they match
print(f"\nDo they match? {result[0] == 'ﬁ test'.title()}")

# Let's also test the assertion from the bug report
try:
    assert result[0] == 'Fi Test'
    print("Assertion passed")
except AssertionError:
    print(f"Assertion failed: {result[0]!r} != 'Fi Test'")