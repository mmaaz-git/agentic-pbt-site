from pandas.core.indexers import length_of_indexer

# Test the specific case mentioned in the report
print("Testing range(0, 1, 2):")
indexer = range(0, 1, 2)
result = length_of_indexer(indexer)
expected = len(list(indexer))
print(f"  Result: {result}")
print(f"  Expected: {expected}")
print(f"  List of range: {list(indexer)}")
print()

# Test all the cases from the report
test_cases = [
    (0, 1, 2),
    (0, 5, 3),
    (0, 10, 7),
    (5, 10, 1),
    (0, 0, 1),
]

print("Testing all cases from the report:")
for start, stop, step in test_cases:
    indexer = range(start, stop, step)
    result = length_of_indexer(indexer)
    expected = len(list(indexer))
    match = "✓" if result == expected else "✗"
    print(f"{match} range({start}, {stop}, {step}): computed={result}, expected={expected}, actual_list={list(indexer)}")

print()

# Test some edge cases
print("Testing additional edge cases:")
edge_cases = [
    (1, 0, 1),    # Empty range (start > stop)
    (0, 10, 100), # Step larger than range
    (10, 20, 2),  # Even step
    (10, 20, 3),  # Odd step
    (0, 100, 7),  # Larger range
]

for start, stop, step in edge_cases:
    indexer = range(start, stop, step)
    result = length_of_indexer(indexer)
    expected = len(list(indexer))
    match = "✓" if result == expected else "✗"
    print(f"{match} range({start}, {stop}, {step}): computed={result}, expected={expected}")