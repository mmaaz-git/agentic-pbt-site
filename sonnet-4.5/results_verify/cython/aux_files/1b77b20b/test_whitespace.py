from Cython.Build.Dependencies import parse_list

# Test cases from the bug report
test_cases = [
    ("[' ']", [' ']),  # quoted whitespace in bracket notation
    ('a " " b', ['a', ' ', 'b']),  # quoted whitespace in space-delimited format
    ("['   ']", ['   ']),  # multiple spaces
    ("['\t']", ['\t']),  # tab character
    ("['\\n']", ['\\n']),  # newline (escaped in the string)
]

print("Testing parse_list with whitespace:")
print("-" * 40)

for input_str, expected in test_cases:
    result = parse_list(input_str)
    match = result == expected
    print(f"Input:    {repr(input_str)}")
    print(f"Expected: {expected}")
    print(f"Got:      {result}")
    print(f"Match:    {match}")
    if not match:
        print("FAIL!")
    print()

# Also test the doctests
print("\nDoctest examples:")
print("-" * 40)

doctest_cases = [
    ("", []),
    ("a", ['a']),
    ("a b c", ['a', 'b', 'c']),
    ("[a, b, c]", ['a', 'b', 'c']),
    ('a " " b', ['a', ' ', 'b']),
    ('[a, ",a", "a,", ",", ]', ['a', ',a', 'a,', ',']),
]

for input_str, expected in doctest_cases:
    result = parse_list(input_str)
    match = result == expected
    print(f"Input:    {repr(input_str)}")
    print(f"Expected: {expected}")
    print(f"Got:      {result}")
    print(f"Match:    {match}")
    if not match:
        print("FAIL!")
    print()