import numpy as np
import numpy.char as char

# Test various strings with null characters
test_cases = [
    '',           # Empty string
    'a',          # String without null
    '\x00',       # Just null character
    'a\x00b',     # String with null in middle
    '\x00ab',     # String starting with null
    'ab\x00',     # String ending with null
]

for test_str in test_cases:
    arr = np.array([test_str])

    print(f"\nTesting string: {test_str!r}")
    print("find('\\x00'):")
    numpy_result = char.find(arr, '\x00')[0]
    python_result = test_str.find('\x00')
    print(f"  numpy: {numpy_result}, python: {python_result}, match: {numpy_result == python_result}")

    print("rfind('\\x00'):")
    numpy_result = char.rfind(arr, '\x00')[0]
    python_result = test_str.rfind('\x00')
    print(f"  numpy: {numpy_result}, python: {python_result}, match: {numpy_result == python_result}")

    print("startswith('\\x00'):")
    numpy_result = char.startswith(arr, '\x00')[0]
    python_result = test_str.startswith('\x00')
    print(f"  numpy: {numpy_result}, python: {python_result}, match: {numpy_result == python_result}")

    print("endswith('\\x00'):")
    numpy_result = char.endswith(arr, '\x00')[0]
    python_result = test_str.endswith('\x00')
    print(f"  numpy: {numpy_result}, python: {python_result}, match: {numpy_result == python_result}")