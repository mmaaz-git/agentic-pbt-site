#!/usr/bin/env python3
"""Test strip_string_literals behavior with empty strings"""

from Cython.Build.Dependencies import strip_string_literals

def test_strip(input_str, desc):
    print(f"\n{desc}")
    print(f"Input: {repr(input_str)}")
    result, literals = strip_string_literals(input_str)
    print(f"Result: {repr(result)}")
    print(f"Literals dict: {literals}")
    return result, literals

# Test various cases
test_strip('""', "Empty double-quoted string")
test_strip("''", "Empty single-quoted string")
test_strip('"a"', "Non-empty double-quoted string")
test_strip('[""]', "Empty string in brackets")
test_strip('["a"]', "Non-empty string in brackets")
test_strip('[a, "", b]', "Mixed with empty string")
test_strip('"', "Single quote character")
test_strip("'", "Single apostrophe")

# Now test the interaction with unquote
print("\n" + "=" * 60)
print("TESTING PARSE_LIST INTERNALS")
print("=" * 60)

test_input = '["a", "", "b"]'
s, literals = strip_string_literals(test_input)
print(f"\nOriginal: {repr(test_input)}")
print(f"After strip: {repr(s)}")
print(f"Literals: {literals}")

# Simulate what happens in parse_list
if len(s) >= 2 and s[0] == '[' and s[-1] == ']':
    s_inner = s[1:-1]
    delimiter = ','
else:
    s_inner = s
    delimiter = ' '

print(f"After bracket removal: {repr(s_inner)}")
print(f"Split by '{delimiter}': {s_inner.split(delimiter)}")

for item in s_inner.split(delimiter):
    item_stripped = item.strip()
    print(f"\nItem: {repr(item_stripped)}")
    if item_stripped and item_stripped[0] in "'\"":
        key = item_stripped[1:-1]
        print(f"  Looking up key: {repr(key)}")
        print(f"  Key in literals? {key in literals}")
        if key in literals:
            print(f"  Value: {repr(literals[key])}")