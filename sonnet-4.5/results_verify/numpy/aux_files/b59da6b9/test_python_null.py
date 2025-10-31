"""Test how Python handles null characters in strings"""

# Test Python's built-in len() with null characters
test_strings = [
    '\x00',
    'a\x00',
    'abc\x00',
    '\x00\x00',
    'abc\x00def'
]

print("Python's handling of null characters in strings:")
print("-" * 50)
for s in test_strings:
    print(f"String: {s!r:<15} | len(): {len(s)}")

print("\nTesting string operations with null characters:")
# Test that null is indeed a valid character
s = 'hello\x00world'
print(f"Original string: {s!r}")
print(f"Length: {len(s)}")
print(f"Index of null: {s.index('\x00')}")
print(f"Slice before null: {s[:5]!r}")
print(f"Slice after null: {s[6:]!r}")

# Test that Python strings can contain multiple nulls
multi_null = '\x00\x00\x00'
print(f"\nString with 3 nulls: {multi_null!r}")
print(f"Length: {len(multi_null)}")
print(f"Count of nulls: {multi_null.count('\x00')}")

# Confirm null is a valid Unicode character
print(f"\nOrd of null character: {ord('\x00')}")
print(f"Chr(0) gives: {chr(0)!r}")