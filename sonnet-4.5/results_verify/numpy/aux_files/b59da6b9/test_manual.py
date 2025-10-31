import numpy.char as char
import numpy as np

test_cases = ['\x00', 'a\x00', 'abc\x00', '\x00\x00']

print("Testing numpy.char.str_len with strings containing null characters:")
print("-" * 60)
print(f"{'String':<15} | {'Python len':<11} | {'numpy str_len':<13} | Match?")
print("-" * 60)

for s in test_cases:
    arr = np.array([s])
    numpy_len = char.str_len(arr)[0]
    python_len = len(s)
    match = "✓" if numpy_len == python_len else "✗"
    print(f'{s!r:<15} | {python_len:<11} | {numpy_len:<13} | {match}')

# Test with null character in the middle (mentioned in the report)
print("\nAdditional test - null in middle:")
test_middle = 'abc\x00def'
arr = np.array([test_middle])
numpy_len = char.str_len(arr)[0]
python_len = len(test_middle)
match = "✓" if numpy_len == python_len else "✗"
print(f'{test_middle!r:<15} | {python_len:<11} | {numpy_len:<13} | {match}')