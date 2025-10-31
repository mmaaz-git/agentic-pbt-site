from Cython.Build.Dependencies import parse_list, strip_string_literals

# Let's see what strip_string_literals does
test_string = "a b # comment"
stripped, literals = strip_string_literals(test_string)
print(f"Original: {test_string!r}")
print(f"Stripped: {stripped!r}")
print(f"Literals: {literals}")

# Now let's see what parse_list does with it
print("\nParsing the original string:")
result = parse_list(test_string)
print(f"Result: {result}")

# Let's also test with a list-style format
test_string2 = "[a, b] # comment"
print(f"\nTest with list format: {test_string2!r}")
result2 = parse_list(test_string2)
print(f"Result: {result2}")

# Let's manually follow the logic of parse_list
test_string3 = "a b # comment"
print(f"\nManual tracing of parse_list for: {test_string3!r}")
s = test_string3
if len(s) >= 2 and s[0] == '[' and s[-1] == ']':
    s = s[1:-1]
    delimiter = ','
else:
    delimiter = ' '
print(f"Delimiter: {delimiter!r}")

s, literals = strip_string_literals(s)
print(f"After stripping: {s!r}")
print(f"Literals dict: {literals}")

def unquote(literal):
    literal = literal.strip()
    if literal[0] in "'\"":
        return literals[literal[1:-1]]
    else:
        return literal

items = s.split(delimiter)
print(f"Split items: {items}")

result = []
for item in items:
    if item.strip():
        unquoted = unquote(item)
        result.append(unquoted)
        print(f"  {item!r} -> {unquoted!r}")

print(f"Final result: {result}")