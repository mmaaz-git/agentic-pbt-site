from Cython.Build.Dependencies import parse_list

# Test case that demonstrates the bug
result = parse_list("a b # comment")
print(f"Result: {result}")
print(f"Expected: ['a', 'b']")

# This assertion will fail, demonstrating the bug
assert result == ['a', 'b'], f"Expected ['a', 'b'], got {result}"