from pandas.core.indexers import length_of_indexer

test_cases = [
    (slice(2, 1, 1), list(range(10))),  # Expected: 0
    (slice(4, 2, 1), list(range(10))),  # Expected: 0
    (slice(None, None, -1), list(range(5))),  # Expected: 5
    (slice(2, 4, -1), list(range(10))),  # Expected: 0
]

print("Testing additional cases mentioned in the bug report:")
for slc, target in test_cases:
    calculated_length = length_of_indexer(slc, target)
    actual_slice = target[slc]
    actual_length = len(actual_slice)

    match = "✓" if calculated_length == actual_length else "✗"
    print(f"{match} slice{slc.start, slc.stop, slc.step} on target len={len(target)}: calculated={calculated_length}, actual={actual_length}, slice={actual_slice}")