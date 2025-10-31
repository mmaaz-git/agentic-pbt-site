import sys

# Test Python's handling of null bytes in str.join
print("Testing Python's str.join with null bytes:")
print("=" * 50)

# Basic tests
test_cases = [
    ('abc', '\x00'),
    ('00', '\x00'),
    ('hello', '\x00'),
    (['a', 'b', 'c'], '\x00'),
    (['1', '2', '3'], '\x00'),
]

for seq, sep in test_cases:
    if isinstance(seq, str):
        # When seq is a string, join treats it as an iterable of characters
        result = sep.join(seq)
        print(f"{sep!r}.join({seq!r}) = {result!r}")
    else:
        # When seq is a list
        result = sep.join(seq)
        print(f"{sep!r}.join({seq!r}) = {result!r}")

# Verify null bytes are preserved in Python strings
print("\nNull bytes are preserved in Python strings:")
test_str = 'a\x00b\x00c'
print(f"test_str = {test_str!r}")
print(f"len(test_str) = {len(test_str)}")
print(f"list(test_str) = {list(test_str)}")

# Confirm that Python strings can contain null bytes
print("\nPython strings can contain null bytes:")
s = '\x00'.join(['x', 'y', 'z'])
print(f"'\\x00'.join(['x', 'y', 'z']) = {s!r}")
print(f"Length: {len(s)} (should be 5)")
print(f"Bytes: {s.encode('utf-8')}")