import numpy as np
import numpy.strings as nps

# Test various cases with null character searching
test_cases = [
    '',
    'abc',
    'a\x00b',
    '\x00\x00',
    'hello world',
    '\x00',
    'a',
    'abc\x00',
    '\x00abc',
    'a\x00b\x00c'
]

print("Testing numpy.strings.rfind with null character ('\\x00'):")
print("=" * 70)
print(f"{'String':<20} | {'Python rfind':<15} | {'NumPy rfind':<15} | {'Match?':<10}")
print("-" * 70)

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_rfind = nps.rfind(arr, '\x00')[0]
    py_rfind = s.rfind('\x00')
    match = "✓" if np_rfind == py_rfind else "✗"

    # Format string representation for display
    s_repr = repr(s) if s else "''"
    print(f"{s_repr:<20} | {py_rfind:<15} | {np_rfind:<15} | {match:<10}")

print("\n" + "=" * 70)
print("\nKey observations:")
print("1. When '\\x00' is NOT in string: NumPy returns len(string) instead of -1")
print("2. When '\\x00' IS in string: NumPy often returns wrong position")
print("3. Python's str.rfind correctly returns -1 when not found")