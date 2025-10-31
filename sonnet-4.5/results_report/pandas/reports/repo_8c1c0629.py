from pandas.core.indexers import length_of_indexer

# Test case from the bug report
r = range(0, 1, 2)
expected = len(r)
predicted = length_of_indexer(r)

print(f"Range object: {r}")
print(f"Expected length (using len()): {expected}")
print(f"Predicted length (using length_of_indexer): {predicted}")
print(f"Bug present: {expected != predicted}")
print()

# Additional test cases showing the pattern of the bug
test_cases = [
    range(0, 1, 2),   # step > (stop - start)
    range(0, 1, 3),   # step >> (stop - start)
    range(0, 5, 10),  # another case where step > (stop - start)
    range(1, 2, 5),   # non-zero start
    range(0, 10, 1),  # normal case - should work
    range(0, 10, 2),  # normal case - should work
    range(0, 10, 3),  # edge case where (10 - 0) // 3 = 3 but len() = 4
    range(0, 9, 3),   # case where it works correctly
]

print("Testing various range objects:")
print("-" * 60)
for r in test_cases:
    expected = len(r)
    predicted = length_of_indexer(r)
    status = "✓ OK" if expected == predicted else "✗ BUG"
    print(f"{str(r):20} | len()={expected:2} | length_of_indexer()={predicted:2} | {status}")