from pandas.core.indexers.utils import length_of_indexer

target = list(range(50))
slc = slice(1, 0, None)

computed = length_of_indexer(slc, target)
actual = len(target[slc])

print(f"length_of_indexer(slice(1, 0, None), target) = {computed}")
print(f"len(target[slice(1, 0, None)]) = {actual}")
print(f"Are they equal? {computed == actual}")

# Additional test cases
test_cases = [
    (slice(1, 0, None), "Empty slice (start > stop)"),
    (slice(5, 2, None), "Another empty slice"),
    (slice(0, 0, None), "Start equals stop"),
    (slice(0, 5, None), "Normal slice"),
]

for slc, description in test_cases:
    computed = length_of_indexer(slc, target)
    actual = len(target[slc])
    print(f"\n{description}: {slc}")
    print(f"  length_of_indexer: {computed}")
    print(f"  len(target[slc]): {actual}")
    print(f"  Match: {computed == actual}")