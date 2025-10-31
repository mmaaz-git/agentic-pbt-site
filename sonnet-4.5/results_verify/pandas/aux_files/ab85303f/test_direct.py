from pandas.core.indexers import length_of_indexer

rng = range(1, 0, 1)

expected_length = len(rng)
predicted_length = length_of_indexer(rng)

print(f"Expected length: {expected_length}")
print(f"Predicted length: {predicted_length}")
print(f"Match: {expected_length == predicted_length}")

# Test a few more cases to understand the pattern
test_cases = [
    range(1, 0, 1),   # Empty range (start > stop, positive step)
    range(0, 1, 1),   # Normal range with 1 element
    range(5, 3, 1),   # Empty range (start > stop, positive step)
    range(10, 10, 1), # Empty range (start == stop)
    range(0, 5, 2),   # Normal range with step 2
    range(0, -5, -1), # Normal range with negative step
    range(-5, 0, 1),  # Normal range with negative start
]

print("\nTesting various range cases:")
for r in test_cases:
    expected = len(r)
    predicted = length_of_indexer(r)
    match = "✓" if expected == predicted else "✗"
    print(f"{match} range{r.start, r.stop, r.step}: expected={expected}, predicted={predicted}")