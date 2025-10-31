import sys

# Test Python's built-in string multiplication with null characters
test_cases = [
    ('\x00', 3),
    ('\x00\x00', 2),
    ('a\x00', 2),
    ('a\x00b', 2),
]

print("Python's built-in string multiplication behavior:")
for s, n in test_cases:
    result = s * n
    print(f"  {repr(s):10} * {n} = {repr(result)}")
    print(f"    Length: input={len(s)}, output={len(result)}, expected={len(s)*n}")

print("\nPython version:", sys.version)