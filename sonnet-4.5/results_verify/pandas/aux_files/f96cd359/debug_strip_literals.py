#!/usr/bin/env python3

# Debug the strip_string_literals function to understand the bug

from Cython.Build.Dependencies import strip_string_literals, parse_list

# Test with a single quote
test_input = '"'
print(f"Testing with input: {repr(test_input)}")

# Strip string literals
new_code, literals = strip_string_literals(test_input)
print(f"new_code: {repr(new_code)}")
print(f"literals: {literals}")

# Now test with the full parse_list flow
print("\n--- Full parse_list flow ---")
s = test_input
print(f"Input: {repr(s)}")

# Check initial condition for list syntax
if len(s) >= 2 and s[0] == '[' and s[-1] == ']':
    s = s[1:-1]
    delimiter = ','
else:
    delimiter = ' '

print(f"Delimiter: {repr(delimiter)}")

# Strip literals
s, literals = strip_string_literals(s)
print(f"After strip_string_literals:")
print(f"  s: {repr(s)}")
print(f"  literals: {literals}")

# Split and process
items = s.split(delimiter)
print(f"After split: {items}")

# Try to unquote each item
for item in items:
    if item.strip():
        literal = item.strip()
        print(f"\nProcessing item: {repr(literal)}")
        if literal[0] in "'\"":
            print(f"  It starts with a quote")
            key = literal[1:-1]
            print(f"  Key to lookup: {repr(key)}")
            if key in literals:
                print(f"  Found in literals: {literals[key]}")
            else:
                print(f"  NOT found in literals!")
                print(f"  Available keys: {list(literals.keys())}")

# Test with other inputs
print("\n\n=== Testing with other inputs ===")
for test in ["'", '""', "''", '"hello"', "'world'", '"a" "b"', "' '", '" "']:
    print(f"\nInput: {repr(test)}")
    try:
        result = parse_list(test)
        print(f"  Success: {result}")
    except KeyError as e:
        print(f"  KeyError: {e}")
        s, lits = strip_string_literals(test)
        print(f"  strip_string_literals returned: s={repr(s)}, literals={lits}")