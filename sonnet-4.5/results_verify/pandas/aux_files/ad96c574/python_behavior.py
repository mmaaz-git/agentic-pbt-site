"""Test Python's standard behavior with null characters in strings"""

# Test Python's handling of null characters
test_strings = [
    'abc\x00',
    'abc\x00def',
    '\x00abc',
    'a\x00b\x00c',
    '\x000'
]

print("Python's standard string slicing behavior with null characters:")
print("=" * 60)

for s in test_strings:
    print(f"\nOriginal string: {repr(s)}")
    print(f"Length: {len(s)}")
    print(f"Slice [0:]: {repr(s[0:])}")
    print(f"Slice [:]: {repr(s[:])}")
    print(f"Slice [0:len(s)]: {repr(s[0:len(s)])}")
    if len(s) > 3:
        print(f"Slice [0:4]: {repr(s[0:4])}")
    print(f"Characters: {[repr(c) for c in s]}")

print("\n" + "=" * 60)
print("\nKey observation: Python strings preserve ALL characters including nulls")
print("Null characters (\\x00) are valid characters in Python strings")
print("They do not terminate the string like in C")