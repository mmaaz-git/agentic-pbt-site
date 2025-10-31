from pandas.core.indexers.utils import length_of_indexer

# Demonstrate the bug with slice(1, 0, None)
target = list(range(50))
slc = slice(1, 0, None)

# Get the result from length_of_indexer
computed = length_of_indexer(slc, target)

# Get the actual result from Python's built-in slicing
actual = len(target[slc])

print(f"Demonstrating bug with empty slice where start > stop:")
print(f"Target: list(range(50))")
print(f"Slice: slice(1, 0, None)")
print()
print(f"length_of_indexer(slice(1, 0, None), target) = {computed}")
print(f"len(target[slice(1, 0, None)]) = {actual}")
print()
print(f"Expected: {actual}")
print(f"Got: {computed}")
print()

# Test additional cases to show pattern
test_cases = [
    slice(5, 2, None),
    slice(10, 3, None),
    slice(20, 0, None),
    slice(-3, -13, None),
]

print("Additional failing cases:")
for slc in test_cases:
    computed = length_of_indexer(slc, target)
    actual = len(target[slc])
    print(f"  slice{slc.start, slc.stop, slc.step}: computed={computed}, actual={actual}")

print()
print("Assertion check (will fail):")
slc = slice(1, 0, None)
computed = length_of_indexer(slc, target)
actual = len(target[slc])
assert computed == actual, f"Assertion failed: {computed} != {actual}"