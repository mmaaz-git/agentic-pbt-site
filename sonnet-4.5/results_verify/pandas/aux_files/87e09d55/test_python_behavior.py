import sys

# Test various slicing scenarios on empty list
empty_list = []

test_cases = [
    slice(0, -20, None),
    slice(0, -1, None),
    slice(1, None, -1),
    slice(-5, -10, None),
    slice(-20, -1, None),
]

for s in test_cases:
    result = empty_list[s]
    print(f"empty_list[{s}] = {result}, len = {len(result)}")

# Test on non-empty list for comparison
normal_list = [0, 1, 2, 3, 4]
print("\nFor comparison with [0, 1, 2, 3, 4]:")
for s in test_cases:
    result = normal_list[s]
    print(f"list[{s}] = {result}, len = {len(result)}")

# Python's behavior with negative indices
print("\nPython's indexing rules:")
print("When list is empty (len=0):")
print("  slice(0, -20) -> slice(0, 0 + (-20)) -> slice(0, -20)")
print("  Since -20 < 0, this is effectively slice(0, 0) -> empty list")
print("  Python ensures slices always return valid subsequences, never negative lengths")