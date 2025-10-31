from Cython.Build.Dependencies import parse_list

# Test the specific failing input reported in the bug
try:
    result = parse_list('"')
    print(f"Unexpected success: {result}")
except KeyError as e:
    print(f"KeyError as reported: {e}")
except Exception as e:
    print(f"Different error: {type(e).__name__}: {e}")

# Let's also test with the helper function to see what's happening
from Cython.Build.Dependencies import strip_string_literals

# Test what strip_string_literals returns
stripped, literals = strip_string_literals('"')
print(f"strip_string_literals('{chr(34)}') returns:")
print(f"  stripped: {repr(stripped)}")
print(f"  literals: {literals}")

# Now let's manually test the unquote logic
def manual_unquote(literal, literals_dict):
    literal = literal.strip()
    if literal and literal[0] in "'\"":
        key = literal[1:-1]
        print(f"  Trying to lookup key: {repr(key)}")
        print(f"  Available keys: {list(literals_dict.keys())}")
        return literals_dict[key]  # This will fail!
    else:
        return literal

# Test with the result from parse_list
s = '"'
s_stripped, lits = strip_string_literals(s)
print(f"\nManual test of unquote logic:")
for item in s_stripped.split(','):
    if item.strip():
        print(f"  Processing item: {repr(item.strip())}")
        try:
            manual_unquote(item, lits)
        except KeyError as e:
            print(f"  KeyError: {e} - key not found in literals dict")