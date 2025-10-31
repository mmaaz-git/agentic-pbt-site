from pandas.core.indexers import length_of_indexer

# Test case 1: Empty range where start > stop (should be 0, not negative)
rng1 = range(1, 0, 1)
expected_length1 = len(rng1)
predicted_length1 = length_of_indexer(rng1)

print(f"Test case 1: range(1, 0, 1)")
print(f"  Expected length: {expected_length1}")
print(f"  Predicted length: {predicted_length1}")
print(f"  Match: {expected_length1 == predicted_length1}")
print()

# Test case 2: Another empty range
rng2 = range(5, 3, 1)
expected_length2 = len(rng2)
predicted_length2 = length_of_indexer(rng2)

print(f"Test case 2: range(5, 3, 1)")
print(f"  Expected length: {expected_length2}")
print(f"  Predicted length: {predicted_length2}")
print(f"  Match: {expected_length2 == predicted_length2}")
print()

# Test case 3: Non-empty range with step > 1
rng3 = range(0, 5, 2)
expected_length3 = len(rng3)
predicted_length3 = length_of_indexer(rng3)

print(f"Test case 3: range(0, 5, 2)")
print(f"  Expected length: {expected_length3}")
print(f"  Predicted length: {predicted_length3}")
print(f"  Match: {expected_length3 == predicted_length3}")
print()

# Test case 4: Normal range with step = 1
rng4 = range(0, 5, 1)
expected_length4 = len(rng4)
predicted_length4 = length_of_indexer(rng4)

print(f"Test case 4: range(0, 5, 1)")
print(f"  Expected length: {expected_length4}")
print(f"  Predicted length: {predicted_length4}")
print(f"  Match: {expected_length4 == predicted_length4}")