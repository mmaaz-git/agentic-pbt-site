from pandas.core.indexers import length_of_indexer

# Test case 1: Simple list with negative step
slc = slice(None, None, -1)
target = [0]

expected = length_of_indexer(slc, target)
actual = len(target[slc])

print(f"Test 1: slice(None, None, -1) on [0]")
print(f"length_of_indexer: {expected}")
print(f"actual: {actual}")
print(f"target[slc]: {target[slc]}")
print()

# Test case 2: Longer list with negative step
target2 = [0, 1, 2, 3, 4]
slc2 = slice(None, None, -1)

expected2 = length_of_indexer(slc2, target2)
actual2 = len(target2[slc2])

print(f"Test 2: slice(None, None, -1) on [0, 1, 2, 3, 4]")
print(f"length_of_indexer: {expected2}")
print(f"actual: {actual2}")
print(f"target[slc]: {target2[slc2]}")
print()

# Test case 3: With step -2
slc3 = slice(None, None, -2)

expected3 = length_of_indexer(slc3, target2)
actual3 = len(target2[slc3])

print(f"Test 3: slice(None, None, -2) on [0, 1, 2, 3, 4]")
print(f"length_of_indexer: {expected3}")
print(f"actual: {actual3}")
print(f"target[slc]: {target2[slc3]}")