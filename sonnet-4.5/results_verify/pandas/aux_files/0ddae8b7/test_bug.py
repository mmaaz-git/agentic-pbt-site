#!/usr/bin/env python3

import pandas.core.indexers as indexers

# Test case from bug report
target = [0, 1, 2, 3, 4]
slc = slice(None, None, -1)

computed = indexers.length_of_indexer(slc, target)
actual = len(target[slc])

print(f"Test case: slice(None, None, -1) on list of length 5")
print(f"Computed length: {computed}")
print(f"Actual length: {actual}")
print(f"Actual result: {target[slc]}")
print(f"Match: {computed == actual}")
print()

# Let's test a few more cases to understand the pattern
test_cases = [
    (slice(None, None, -1), [0, 1, 2, 3, 4]),
    (slice(None, None, -2), [0, 1, 2, 3, 4]),
    (slice(4, None, -1), [0, 1, 2, 3, 4]),
    (slice(None, 0, -1), [0, 1, 2, 3, 4]),
    (slice(4, 0, -1), [0, 1, 2, 3, 4]),
    (slice(None, None, 1), [0, 1, 2, 3, 4]),
    (slice(2, None, -1), [0, 1, 2, 3]),
]

print("Additional test cases:")
for slc, target in test_cases:
    computed = indexers.length_of_indexer(slc, target)
    actual = len(target[slc])
    print(f"slice({slc.start}, {slc.stop}, {slc.step}) on {target}")
    print(f"  Computed: {computed}, Actual: {actual}, Match: {computed == actual}")
    if computed != actual:
        print(f"  ERROR: Mismatch! target[slc] = {target[slc]}")
    print()