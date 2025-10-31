from pandas.core.indexers.utils import length_of_indexer

# Test case 1: Basic negative step slice that should reverse a list
target = [0]  # Single element list
indexer = slice(None, None, -1)  # Standard Python idiom for reversing

claimed_length = length_of_indexer(indexer, target)
actual_result = target[indexer]  # This would be [0] in normal Python
actual_length = len(actual_result)

print("Test case 1: Single element list with reverse slice")
print(f"Target: {target}")
print(f"Indexer: {indexer}")
print(f"Actual result of target[indexer]: {actual_result}")
print(f"Actual length: {actual_length}")
print(f"Claimed length from length_of_indexer: {claimed_length}")
print(f"Match: {actual_length == claimed_length}")
print()

# Test case 2: Multi-element list with reverse slice
target2 = [0, 1, 2, 3, 4]
indexer2 = slice(None, None, -1)

claimed_length2 = length_of_indexer(indexer2, target2)
actual_result2 = target2[indexer2]
actual_length2 = len(actual_result2)

print("Test case 2: Multi-element list with reverse slice")
print(f"Target: {target2}")
print(f"Indexer: {indexer2}")
print(f"Actual result of target[indexer]: {actual_result2}")
print(f"Actual length: {actual_length2}")
print(f"Claimed length from length_of_indexer: {claimed_length2}")
print(f"Match: {actual_length2 == claimed_length2}")
print()

# Test case 3: Step of -2
target3 = [0, 1, 2, 3, 4]
indexer3 = slice(None, None, -2)

claimed_length3 = length_of_indexer(indexer3, target3)
actual_result3 = target3[indexer3]
actual_length3 = len(actual_result3)

print("Test case 3: Multi-element list with step=-2")
print(f"Target: {target3}")
print(f"Indexer: {indexer3}")
print(f"Actual result of target[indexer]: {actual_result3}")
print(f"Actual length: {actual_length3}")
print(f"Claimed length from length_of_indexer: {claimed_length3}")
print(f"Match: {actual_length3 == claimed_length3}")