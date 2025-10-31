import numpy.char as char
import numpy as np

test_cases = ['\x00', 'a\x00', 'abc\x00', '\x00\x00', 'abc\x00def']

print("Testing numpy.char.str_len with null characters:")
print("=" * 60)
print(f"{'String':20} | {'Python len':10} | {'numpy str_len':13} | {'Match?':7}")
print("-" * 60)

for s in test_cases:
    arr = np.array([s])
    numpy_len = char.str_len(arr)[0]
    python_len = len(s)
    match = "✓" if numpy_len == python_len else "✗"
    print(f'{repr(s):20} | {python_len:10} | {numpy_len:13} | {match:7}')

print("\nDetailed analysis:")
print("-" * 60)
print("Pattern: numpy.char.str_len() stops counting at trailing null characters")
print("but correctly handles null characters in the middle of strings.")