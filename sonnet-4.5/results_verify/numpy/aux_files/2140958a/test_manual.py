import numpy as np
import numpy.char

# Test case from the bug report
s = 'test\x00'
arr = np.array([s])
result = numpy.char.multiply(arr, 1)

print(f'Input: {s!r} (len={len(s)})')
print(f'Expected: {s * 1!r} (len={len(s * 1)})')
print(f'Actual: {result[0]!r} (len={len(result[0])})')

try:
    assert result[0] == s
    print("Assertion passed")
except AssertionError:
    print("AssertionError: result does not match expected value")

# Additional tests to understand the behavior
print("\n--- Additional tests ---")

# Test with different null positions
test_cases = [
    '\x00test',      # null at beginning
    'te\x00st',      # null in middle
    'test\x00',      # null at end
    '\x00\x00',      # multiple nulls
    'test\x00\x00',  # multiple trailing nulls
    '',              # empty string
    'test'           # no nulls
]

for test_str in test_cases:
    arr = np.array([test_str])
    result = numpy.char.multiply(arr, 1)
    print(f"Input: {test_str!r} (len={len(test_str)}) -> Result: {result[0]!r} (len={len(result[0])})")