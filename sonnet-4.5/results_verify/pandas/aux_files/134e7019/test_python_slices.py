"""Test Python's built-in slice behavior with negative indices"""

# Test cases to understand Python's slice behavior with negative indices
test_cases = [
    # Empty lists with negative indices
    ([], slice(None, -1, None)),
    ([], slice(None, -2, None)),
    ([], slice(None, -10, None)),
    ([], slice(-1, None, None)),
    ([], slice(-2, None, None)),

    # Single element list
    ([1], slice(None, -1, None)),
    ([1], slice(None, -2, None)),
    ([1], slice(-1, None, None)),
    ([1], slice(-2, None, None)),

    # Two element list
    ([1, 2], slice(None, -1, None)),
    ([1, 2], slice(None, -2, None)),
    ([1, 2], slice(None, -3, None)),
    ([1, 2], slice(-1, None, None)),
    ([1, 2], slice(-2, None, None)),
    ([1, 2], slice(-3, None, None)),

    # Longer list
    ([1, 2, 3, 4, 5], slice(None, -1, None)),
    ([1, 2, 3, 4, 5], slice(None, -3, None)),
    ([1, 2, 3, 4, 5], slice(None, -10, None)),
]

print("Python's built-in slice behavior:")
print("=" * 60)

for lst, slc in test_cases:
    result = lst[slc]
    length = len(result)
    print(f"list={lst}, slice={slc}")
    print(f"  Result: {result}, Length: {length}")
    print()

# Key observation: Python ALWAYS returns a valid list with non-negative length
# Even when the negative index is "before" the start of the list